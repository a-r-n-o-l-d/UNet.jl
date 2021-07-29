struct CropCat
    trms
end

function (cc::CropCat)(x1, x2) # x2 : input
    lo = cc.trms .+ 1
    up = size(x2) .- cc.trms
    chcat(x2[UnitRange.(lo, up)...], x1)
end

utrim(i) = (192 - 2^(i + 3)) ÷ 2^i
"""
    unet(; inchannels)
Build a [U-Net](https://arxiv.org/abs/1505.04597v1) to process images with 
`inchannels` channels (`inchannels` = 3 for RGB images). This implementation 
corresponds to original paper with unpadded convolutions.
"""
function unet(; inchannels)
    function uconv(ch)
        ich, och = ch
        [Conv((3, 3), ich=>och, relu), Conv((3, 3), och=>och, relu)]
    end
    
    udown() = MaxPool((2, 2))
    
    uup(ch) = ConvTranspose((2, 2), ch, relu, stride = (2, 2))
    
    function downblock(ich)
        och = 2 * ich
        [udown(), uconv(ich=>och)...]
    end
    
    function upblock(ich)
        mch = ich ÷ 2
        och = mch ÷ 2
        [uconv(ich=>mch)..., uup(mch=>och)]
    end

    enc = Any[uconv(inchannels=>64)]
    dec = Any[Chain(uconv(128=>64)..., Conv((1, 1), 64=>1, sigmoid))]
    dch = 64
    uch = 256
    for l ∈ 1:3
        push!(enc, downblock(dch))
        push!(dec, upblock(uch))
        dch *= 2
        uch *= 2
    end
    b = Chain(udown(), uconv(512=>1024)..., uup(1024=>512))
    con = []
    for l ∈ 1:4
        t = utrim(l)
        push!(con, CropCat((t, t)))
    end
    uchain(encoders = enc, decoders = dec, bridge = b, connection = con)
end

const uminsize = 204

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
function upadding(is)
    tr = utrim(1)
    
    function newsize(is)
        n = is + 2 * tr + 8
        k = ceil(Int, (n - uminsize) / 16)
        uminsize + k * 16
    end
    
    pa = (newsize.(is) .- is) ./ 2
    os = (floor.(Int, pa), ceil.(Int, pa))
    os, ([o .- tr .- 4 for o ∈ os]...,)
end