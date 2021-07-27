module UNet

using Flux

export chcat, uchain

"""
    chcat(x1, x2)

Concatenate two arrays along channel dimension. The channel dimension is assumed
to be the penultimate dimension.

# Examples
```jldoctest
julia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels 

julia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> chcat(x1, x2) |> size
(32, 32, 8, 6)
```
"""
chcat(x1, x2) = cat(x1, x2, dims = (x1 |> size |> length) - 1)

"""

"""
function uchain(;input, output, encoders, decoders, bridge, connector)
    length(encoders) == length(decoders) || println("pouet")
    
    connect(enc::T, prl, dec::T) where {T<:Union{Chain, AbstractArray}} =
       Chain(enc..., prl, dec...)

    connect(enc, prl, dec) = Chain(enc, prl, dec)
    
    getconn(i, c::AbstractArray) = c[i]
    
    getconn(i, c) = c
    
    getconn(c::AbstractArray) = c[end]
    
    getconn(c) = c
    
    ite = reverse(eachindex(encoders))
    c = getconn(connector)
    l = SkipConnection(bridge, c)
    for i ∈ ite
        l = connect(encoders[i], l, decoders[i])
        c = getconn(i, connector)
        l = SkipConnection(l, c)
    end
    connect(input, l, output)
end

end
