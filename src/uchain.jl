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
        #utrim(l, nlvl) = - (2^(l + 2) - 3 * 2^(nlvl + 1)) ÷ 2^(l - 1)
        
        #uminsize(nlvl) = 13 * 2^nlvl - 4
        #uminsize(; padding, nlevels) = padding ? 4 * 2^nlevels : 3 * 2^(nlevels + 2) - 4
        
        #=
        
        BUG si is est trop petit
        
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
        
        function upadding(sz; padding, nlevels)
            tr = utrim(1, nlevels)
            ms = uminsize(padding = padding, nlevels = nlevels)
        
            n = @. sz + 2 * tr + 2 * 4 # trimming + 4 unpadded convolutions
            ns = @. ms + ceil(Int, (n - ms) / 16) * 16 # step = 2^nlevels
            for i ∈ eachindex(ns)
                if ns[i] < ms
                    ns[i] = ms
                end
            end
            
            pa = @. (ns - sz) / 2
            ilo = floor.(Int, pa)  # lower edge padding for input
            ihi = ceil.(Int, pa)   # upper edge padding for input
            glo = @. ilo - tr - 4  # lower edge padding for ground truth
            ghi = @. ihi - tr - 4  # upper edge padding for ground truth
            (ilo, ihi), (glo, ghi)
        end
        =#
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

for T ∈ [:Chain :AbstractArray]
    @eval begin
        @inline uconnect(enc::($T), prl, dec) = Chain(enc..., prl, dec)
        @inline uconnect(enc, prl, dec::($T)) = Chain(enc, prl, dec...)
    end
end

@inline ubridge(b, c) = SkipConnection(b, c)

@inline ubridge(b::AbstractArray, c) = SkipConnection(Chain(b...), c)