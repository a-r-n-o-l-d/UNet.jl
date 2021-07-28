module UNet

using Flux

export chcat, uchain

"""
    chcat(x...)

Concatenate the image data along the dimension corresponding to the channels.
Image data should be stored in WHCN order (width, height, channels, batch) or
WHDCN (width, height, depth, channels, batch) in 3D context. Channels are
assumed to be the penultimate dimension.

# Examples
```jldoctest
julia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels 

julia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> chcat(x1, x2) |> size
(32, 32, 8, 6)
```
"""
chcat(x...) = cat(x...; dims = (x1 |> size |> length) - 1)

"""
    uchain(;input, output, encoders, decoders, bridge, connection)

Build a `Chain` with U-Net like architecture.
                                                                                
                                                                                
+---------+                                                          +---------+
|  Input  |                                                          | Output  |
+---------+                                                          +---------+
     |------------------------------------------------------------------->^     
     |                                                                    |     
     |   +---------+                                         +---------+  |     
     +-->|Encoder 1|                                         |Decoder 1|--+     
         +---------+                                         +---------+        
             |--------------------------------------------------->^             
             |                                                    |             
             |   +---------+                         +---------+  |             
             +-->|Encoder 2|                         |Decoder 2|--+             
                 +---------+                         +---------+                
                     |----------------------------------->^                     
                     |                                    |                     
                     |   +---------+         +---------+  |                     
                     +-->|Encoder 3|         |Decoder 3|--+                     
                         +---------+         +---------+                        
                             |------------------->^                             
                             |                    |                             
                             |      +-------+     |                             
                             +----->|Bridge |-----+                             
                                    +-------+                                                                            
"""
function uchain(;input, output, encoders, decoders, bridge, connection)
    length(encoders) == length(decoders) || println("pouet")
    
    connect(enc::T, prl, dec::T) where {T<:Union{Chain, AbstractArray}} =
       Chain(enc..., prl, dec...)

    connect(enc, prl, dec) = Chain(enc, prl, dec)
    
    getconn(i, c::AbstractArray) = c[i]
    
    getconn(i, c) = c
    
    getconn(c::AbstractArray) = c[end]
    
    getconn(c) = c
    
    ite = reverse(eachindex(encoders))
    c = getconn(connection)
    l = SkipConnection(bridge, c)
    for i ∈ ite
        l = connect(encoders[i], l, decoders[i])
        c = getconn(i, connection)
        l = SkipConnection(l, c)
    end
    connect(input, l, output)
end

end
