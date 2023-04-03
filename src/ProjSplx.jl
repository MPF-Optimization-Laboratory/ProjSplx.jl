module ProjSplx

using LinearAlgebra

export projsplx!, projsplx, projnorm1,
       projnorm1!, projnorm1t, proxinf!

"""
Projection onto the unit simplex {x | sum(x) = τ, x ≥ 0}.

   projsplx!(b, τ)

In-place variant of `projsplx`.
"""
function projsplx!(b::Vector{T}, τ::T) where T

    n = length(b)
    bget = false

    idx = sortperm(b, rev=true)
    tsum = zero(T)

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax ≥ b[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        tmax = (tsum + b[idx[n]] - τ) / n
    end

    @inbounds for i = 1:n
        b[i] = max(b[i] - tmax, 0)
    end

end

"""
Projection onto the weighted simplex.

    projsplx!(b, c, τ)

This projects b (in-place) onto the unit simplex weighted by c.
"""
function projsplx!(b::Vector{T}, c::Vector{T}, τ::T) where T

    n = length(b)
    bget = false

    @assert length(b) == length(c) "lengths must match"
    @assert minimum(c) > 0 "c is not positive."

    idx = sortperm(b./c, rev=true)
    tsum = csum = zero(T)

    @inbounds for i = 1:n-1
        j = idx[i]
        tsum += b[j]*c[j]
        csum += c[j]*c[j]
        tmax = (tsum - τ) / csum
        if tmax >= b[idx[i+1]] / c[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        p = idx[n]
        tsum += b[p]*c[p]
        csum += c[p]*c[p]
        tmax = (tsum - τ) / csum
    end

    for i = 1:n
        @inbounds b[i] = max(b[i] - c[i]*tmax, 0)
    end

    return

end

"""
Projection onto the simplex.

  projsplx(b, τ) -> x

Variant of `projsplx`.
"""
function projsplx(b::Vector, τ)
    x = copy(b)
    projsplx!(x, τ)
    return x
end

"""
Projection onto the weighted simplex.

    projsplx(b, c, τ) -> x

Variant of `projsplx!`.
"""
function projsplx(b::Vector, c::Vector, τ)
    x = copy(b)
    projsplx!(x, c, τ)
    return x
end

# s = sign_abs!(x) returns
#     s[i] = true  if x[i] > 0
#     s[i] = false otherwise
# and x = abs(x).
function sign_abs!(x::Vector)
  n = length(x)
  s = Vector{Bool}(undef,n)
  @inbounds for i=1:n
    s[i] = x[i] ≥ 0
    x[i] = abs(x[i])
  end
  return s
end

# set_sign!(x, s) sets the sign of x based on s.
function set_sign!(x::Vector, s::Vector{Bool})
  n = length(x)
  @inbounds for i=1:n
      x[i] = s[i] ? x[i] : -x[i]
  end
end

"""
Projection onto the 1-norm ball

    {x | ||x||₁ ≤ τ }.

In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, τ::Real)
    norm(b,1) > τ || return
    s = sign_abs!(b)
    projsplx!(b, τ)
    set_sign!(b, s)
end

"""
Projection onto the weighted 1-norm ball

    {x | ||diag(c)x||₁ ≤ τ }, c > 0.

In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, c::Vector, τ::Real)
    norm(b,1) > τ || return
    s = sign_abs!(b)
    projsplx!(b, c, τ)
    set_sign!(b, s)
end

"""
Projection onto the weighted 1-norm ball.

Variant of `projnorm1!`.
"""
function projnorm1(b::Vector, c::Vector, τ::Real)
    x = copy(b)
    projnorm1!(x, c, τ)
    return x
end

"""
Projection onto the 1-norm ball.

Variant of `projnorm1!`.
"""
function projnorm1(b::Vector, τ::Real)
    x = copy(b)
    projnorm1!(x, τ)
    return x
end

"""
Proximal map of the scaled infinity norm.
    prox_inf(x,λ) = x - proj(x | λ𝔹₁)
     env_inf(x,λ) = (1/2λ)||x||² - (1/2λ)dist²(x | λ𝔹₁)
Modifies `x` in place; returns the envelope.
"""
function proxinf!(x::Vector, λ::Real)
  λ == 0 && return norm(x, Inf)
  nrmx2 = dot(x,x)
  xp = projnorm1(x, λ)
  BLAS.axpy!(-1., xp, x) # x <- x - xp
  return nrmx2/(2λ) - dot(x,x)/(2λ)
end

"""
Proximal map of the scaled infinity norm.
Return variant of `proxinf!`
"""
function proxinf(x::Vector, λ::Real)
    z = copy(x)
    proxinf!(z, λ)
    return z
end

end # module
