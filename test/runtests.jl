using ProjSplx
using FactCheck

srand(0)

# KKT residuals for projection onto simplex.
function kkt_splx_pass(x, b; c = ones(b), τ=1.0)
    ϵ = eps(1.0)^(2/3)*maxabs(b)
    r = x - b
    γ = dot(x,r)/τ
    z = r - γ*c
    t1 = τ - ϵ ≤ dot(c,x) ≤ τ + ϵ
    t2 = minimum(x) ≥ 0
    t3 = abs(dot(x, z)) ≤ ϵ
    return t1 & t2 & t3
end

# Opt gap for projection onto 1-norm ball.
function kkt_nrm1_pass(x, b; c = ones(b), τ=1.0)
    ϵ = eps(1.0)^(2/3)*maxabs(b)
    r = b - x
    prObj = 0.5*dot(r,r)
    duObj = dot(b,r) - 0.5*dot(r,r) - τ*norm(r./c,Inf)
    fea = norm(c.*x,1) ≤ τ+ϵ
    gap = (prObj - duObj) ≤ ϵ
    return gap & fea
end
    
facts("Projection onto simplex.") do

    context("Unit simplex") do
        n = 110
        τ = rand()
        b = randn(n)

        # in place version
        x = copy(b)
        projsplx!(x, τ)
        @fact kkt_splx_pass(x, b, τ=τ) --> true

        # return version
        x = projsplx(b, τ)
        @fact kkt_splx_pass(x, b, τ=τ) --> true
    end

    context("Weighted unit simplex") do
        n = 101
        τ = rand()
        b = randn(n)
        c = rand(n)

        # in place version
        x = copy(b)
        projsplx!(x, c, τ)
        @fact kkt_splx_pass(x, b, c=c, τ=τ) --> true

        # return version
        x = projsplx(b, c, τ)
        @fact kkt_splx_pass(x, b, c=c, τ=τ) --> true        
    end

end

facts("Projection onto 1-norm ball.") do

    n = 99
    b = randn(n)
    c = rand(n)

    context("1-norm ball (from outside)") do
        τ = 0.1*norm(b,1)

        # in-place version
        x = copy(b)
        projnorm1!(x, τ)
        @fact kkt_nrm1_pass(x, b, τ=τ) --> true

        # return version
        x = projnorm1(b, τ)
        @fact kkt_nrm1_pass(x, b, τ=τ) --> true
    end
        
    context("1-norm ball (from inside)") do
        τ = 1.1*norm(b,1)

        # in-place version
        x = copy(b)
        projnorm1!(x, τ)
        @fact kkt_nrm1_pass(x, b, τ=τ) --> true

        # return version
        x = projnorm1(b, τ)
        @fact kkt_nrm1_pass(x, b, τ=τ) --> true
    end
    
    context("weighted 1-norm ball") do
        τ = 0.1*norm(b,1)

        # in-place version
        x = copy(b)
        projnorm1!(x, c, τ)
        @fact kkt_nrm1_pass(x, b, c=c, τ=τ) --> true

        # return version
        x = projnorm1(b, c, τ)
        @fact kkt_nrm1_pass(x, b, c=c, τ=τ) --> true
        
    end
        
end
