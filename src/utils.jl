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

function checksize(sz::Int; padding, nlevels) # mod(sz, 2^nlevels) == 0
    # number of trimmed pixels due to convolutions
    pl = padding ? 0 : 4
    chk = true
    sz -= pl
    for l ∈ 1:nlevels
        if iseven(sz)
            sz ÷= 2  # MaxPool
            sz -= pl # pixel trimming
        else
            chk = false
            break
        end
    end
    chk
end

checksize(sz::Tuple; kwargs...) = @. checksize(sz; kwargs...)

minsize(; padding, nlevels) = padding ? 4 * 2^nlevels : 3 * 2^(nlevels + 2) - 4

minsize(p, n) = minsize(padding = p, nlevels = n)

trim(l, nlvl) = -(2^(l + 2) - 3 * 2^(nlvl + 1)) ÷ 2^(l - 1)

function outputsize(sz; padding, nlevels)
    if padding
        os = sz
    else
        tr = trim(1, nlevels)
        os = @. sz - 2 * tr - 8
    end
    os
end

outputsize(sz, p, n) = outputsize(sz; padding = p, nlevels = n)

function adjustsize(sz::Int; padding, nlevels)
    # Number of trimmed pixels if unpadded convolutions
    trm = padding ? 0 : (2 * trim(1, nlevels) + 8)
    sz += trm
    # Minimal size required for UNet
    smin =  minsize(padding, nlevels)
    if sz < smin
        ns = smin
    else
        # Ensure that each entry in MaxPool is even
        stp = 2^nlevels
        k = ceil(Int, (sz - smin) / stp)
        ns = smin + k * stp
    end
    ns
end

adjustsize(sz::Int, p, n) = adjustsize(sz, padding = p, nlevels = n)

adjustsize(sz::Tuple, p, n) = adjustsize(sz, padding = p, nlevels = n)

adjustsize(sz::Tuple; kwargs...) = @. adjustsize(sz; kwargs...)

#=Return a tuple of padding values for both input image and ground truth image.
    is : input size
    d : U-Net depth (default 4, as in original implementation)
    nc : number of unpadded convolution per U-block (default 2, as in original implementation)
    
    using Images, ImageIO
    
    ip, op = upadding((256, 256))
    pimg = padarray(img, Pad(:reflect, ip...))
    pgth = padarray(gth, Pad(:reflect, op...))
    =#
function padding(sz; padding, nlevels)
    ns = adjustsize(sz, padding, nlevels)
    pa = @. (ns - sz) / 2
    # lower edge padding for input
    ilo = floor.(Int, pa)
    # upper edge padding for input
    ihi = ceil.(Int, pa)
    # Compute output size
    os = padding ? ns : (@. ns - 2 * trim(1, nlevels) - 8)
    pa = @. (os - sz) / 2
    # lower edge padding for ground truth
    glo = floor.(Int, pa)
    # upper edge padding for ground truth
    ghi = ceil.(Int, pa)
    (ilo, ihi), (glo, ghi)
end