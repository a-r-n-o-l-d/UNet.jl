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

@inline uconnect(enc, prl, dec) = Chain(enc, prl, dec)

for T1 ∈ [:Chain :AbstractArray], T2 ∈ [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc::($T1), prl, dec::($T2)) = Chain(enc..., prl, dec...)
    end
end

for T1 ∈ [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc::($T1), prl, dec) = Chain(enc..., prl, dec)
    end
end

for T2 ∈ [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc, prl, dec::($T2)) = Chain(enc, prl, dec...)
    end
end

@inline ubridge(b, c) = SkipConnection(b, c)

@inline ubridge(b::AbstractArray, c) = SkipConnection(Chain(b...), c)