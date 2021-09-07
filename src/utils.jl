"""
    chcat(x...)

Concatenate the image data along the dimension corresponding to the channels.
Image data should be stored in WHCN order (width, height, channels, batch) or
WHDCN (width, height, depth, channels, batch) in 3D context. Channels are
assumed to be the penultimate dimension.

# Example
```jldoctest
julia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels

julia> chcat(x1, x2) |> size
(32, 32, 8, 6)
```
"""
chcat(x...) = cat(x...; dims = (x[1] |> size |> length) - 1)

uconnect(enc::Chain, prl, dec::Chain) = Chain(enc..., prl, dec...)
    
uconnect(enc::AbstractArray, prl, dec::AbstractArray) = Chain(enc..., prl, dec...)

uconnect(enc::Chain, prl, dec::AbstractArray) = Chain(enc..., prl, dec...)

uconnect(enc::AbstractArray, prl, dec::Chain) = Chain(enc..., prl, dec...)

uconnect(enc, prl, dec) = Chain(enc, prl, dec)

ubridge(b, c) = SkipConnection(b, c)

ubridge(b::AbstractArray, c) = SkipConnection(Chain(b...), c)

"""
    uchain(;encoders, decoders, bridge, connection)

Build a `Chain` with U-Net like architecture. `encoders` and `decoders` are 
arrays of encoding/decoding blocks, from top to bottom (see diagram below). 
`bridge` is the bottom part of U-Net  architecture.
Each level of the U-Net is connected through a 2-argument callable `connection`.
`connection` could be an array in case the way levels are connected vary from 
one level to another.

Notes :
- usually encoder block starts with a 'MaxPool' to downsample image by 2, except
the first level encoder.
- usually decoder block ends with a 'ConvTranspose' to upsample image by 2,
except the first level decoder.

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
              +-->|encoder 3|                        |decoder 3|--+             
                  +---------+                        +---------+                
                       |--------------------------------->^                     
                       |                                  |                     
                       |   +---------+       +---------+  |                     
                       +-->|encoder 4|       |decoder 4|--+                     
                           +---------+       +---------+                        
                                |---------------->^                             
                                |                 |                             
                                |    +--------+   |                             
                                +--->| bridge |---+                             
                                     +--------+                                  
```
See also [`chcat`](@ref).
"""
function uchain(;encoders, decoders, bridge, connection)
    length(encoders) == length(decoders) || throw(ArgumentError(
        "The number of encoders should be equal to the number of decoders."))

    if isa(connection, AbstractArray)
        length(encoders) == length(connection) || throw(ArgumentError(
            "The number of connections should be equal to the number of encoders/decoders."))
    else
        connection = repeat([connection], length(encoders))
    end

    ite = zip(reverse(encoders[2:end]),
              reverse(decoders[2:end]), 
              reverse(connection[1:(end - 1)]))
    l = ubridge(bridge, connection[end])
    for (e, d, c) ∈ ite
        l = uconnect(e, l, d)
        l = SkipConnection(l, c)
    end
    uconnect(encoders[1], l, decoders[1])
end

utrim(l, nlvl) = - (2^(l + 2) - 3 * 2^(nlvl + 1)) ÷ 2^(l - 1)

uminsize(nlvl) = 3 * 2^(nlvl + 2) - 4 + 2^nlvl

#=
Return a tuple of padding values for both input image and ground truth image.
is : input size
d : U-Net depth (default 4, as in original implementation)
nc : number of unpadded convolution per U-block (default 2, as in original implementation)

using Images, ImageIO

ip, op = upadding((256, 256))
pimg = padarray(img, Pad(:reflect, ip...))
pgth = padarray(gth, Pad(:reflect, op...))

=#
function upadding(is, nlvl)
    tr = utrim(1, nlvl)
    ms = uminsize(nlvl)
    
    function newsize(is)
        n = is + 2 * tr + 8
        k = ceil(Int, (n - ms) / 16)
        ms + k * 16
    end
    
    pa = (newsize.(is) .- is) ./ 2
    os = (floor.(Int, pa), ceil.(Int, pa))
    os, ([o .- tr .- 4 for o ∈ os]..., )
end