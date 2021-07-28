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
chcat(x...) = cat(x...; dims = (x[1] |> size |> length) - 1)

"""
    uchain(;input, output, encoders, decoders, bridge, connection)

Build a `Chain` with [U-Net](https://arxiv.org/abs/1505.04597v1) like 
architecture. `encoders` and `decoders` are arrays of encoding/decoding blocks, 
from top to bottom (see diagram below). `bridge` is the bottom part of U-Net 
architecture. 
Each level of the U-Net is connected through a 2-argument callable `connection`.
`connection` could be an array in case the way levels are connected vary from 
one level to another.

Usually encoders start with a 'MaxPool', except the first encoder

```
                                                                                
+---------+                                                          +---------+
|encoder 1|                                                          |decoder 1|
+---------+                                                          +---------+
     |------------------------------------------------------------------->^     
     |                                                                    |     
     |   +---------+                                         +---------+  |     
     +-->|encoder 2|                                         |decoder 2|--+     
         +---------+                                         +---------+        
              |-------------------------------------------------->^             
              |                                                   |             
              |   +---------+                        +---------+  |             
              +--->encoder 3|                        |decoder 3|--+             
                  +---------+                        +---------+                
                       |--------------------------------->^                     
                       |                                  |                     
                       |   +---------+       +---------+  |                     
                       +-->|encoder 4|       |decoder 4|--+                     
                           +---------+       +---------+                        
                                |---------------->^                             
                                |                 |                             
                                |    +-------+    |                             
                                +--->|bridge |----+                             
                                     +-------+                                                                    
```
See also [`chcat`](@ref), [`Flux.SkipConnection`](@ref).
"""
function uchain(;encoders, decoders, bridge, connection)
    length(encoders) == length(decoders) ||
        throw(ArgumentError(
            "The number of encoders should be equal to the number of decoders."))

    if isa(connection, AbstractArray)
        length(encoders) == length(connection) ||
            throw(ArgumentError(
                "The number of connections should be equal to the number of encoders/decoders."))
    else
        connection = repeat([connection], length(encoders))
    end
    
    connect(enc::Chain, prl, dec::Chain) = Chain(enc..., prl, dec...)
    
    connect(enc::AbstractArray, prl, dec::AbstractArray) = Chain(enc..., prl, dec...)
    
    connect(enc::Chain, prl, dec::AbstractArray) = Chain(enc..., prl, dec...)
    
    connect(enc::AbstractArray, prl, dec::Chain) = Chain(enc..., prl, dec...)

    connect(enc, prl, dec) = Chain(enc, prl, dec)

    ite = zip(reverse(encoders[2:end]), reverse(decoders[2:end]), 
              reverse(connection[1:(end - 1)]))
    l = SkipConnection(bridge, connection[end])
    for (e, d, c) ∈ ite
        l = connect(e, l, d)
        l = SkipConnection(l, c)
    end
    connect(encoders[1], l, decoders[1])
end



end
