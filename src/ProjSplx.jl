module ProjSplx

export projsplx!, projsplx, projnorm1, projnorm1!

"""
Projection onto the unit simplex {x | sum(x) = τ, x ≥ 0}.

   projsplx!(b, τ)

In-place variant of `projsplx`.
"""
function projsplx!{T}(b::Vector{T}, τ::T)

    n = length(b)
    bget = false

    idx = sortperm(b, rev=true)
    tsum = zero(T)

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax >= b[idx[i+1]]
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

In-place variant of `projsplx`.
"""
function projsplx!{T}(b::Vector{T}, c::Vector{T}, τ::T)

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

"""
Projection onto the 1-norm ball

    {x | ||x||_1 ≤ τ }.

In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, τ::Real)
    if norm(b,1) ≤ τ
        return
    end
    n = length(b)
    s = Array{Bool}(n)
    @inbounds for i=1:n
        s[i] = b[i] >= 0 ? true : false
        b[i] = abs(b[i])
    end
    projsplx!(b, τ)
    @inbounds for i=1:n
        b[i] = s[i] ? b[i] : -b[i]
    end
end

"""
Projection onto the weighted 1-norm ball

    {x | ||diag(c) x||_1 ≤ τ }, with  c > 0.

In-place variant of `projnorm1`.
"""
function projnorm1!(b::Vector, c::Vector, τ::Real)
    if norm(b,1) ≤ τ
        return
    end
    n = length(b)
    s = Array{Bool}(n)
    @inbounds for i=1:n
        s[i] = b[i] >= 0 ? true : false
        b[i] = abs(b[i])
    end
    projsplx!(b, c, τ)
    @inbounds for i=1:n
        b[i] = s[i] ? b[i] : -b[i]
    end
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

#=
# Specialized 1-norm projection. Not exported in favor of the version above.
function projnorm1(b::Vector, d::Vector; τ::Real = 1.0)

    # Initialization
    n = length(b)
    x = zeros(b)
    
    # Check for quick exit.
    if τ >= norm(d.*b,1)
        return b
    elseif τ < eps(1.0)
        return x
    end

    # Preprocessing (b is assumed to be >= 0)
    idx = sort(b ./ d, rev=true) # Descending.
    permute!(b, idx)
    permute!(d, idx)

    # Optimize
    csdb = csd2 = 0
    soft = alpha1 = 0
    i = 1
    while i <= n
        csdb += d[i].*b[i]
        csd2 += d[i].*d[i]
  
        alpha1 = (csdb - τ) / csd2
        alpha2 = bd[i]

        if alpha1 >= alpha2
            break
        end
        
        soft = alpha1
        i += 1
    end
    x[idx[1:i-1]] = b[1:i-1] - d[1:i-1] * max(0,soft)

    # Restore permutation
    permute!(b, idx)
    permute!(d, idx)

    return x
end
=#

end # module
