struct CropCat
    trms
end

function CropCat(t, vol)
    #t = utrim(lvl)
    if vol
        t = (t, t, t)
    else
        t = (t, t)
    end
    CropCat((t..., 0, 0))
end

function (cc::CropCat)(x1, x2) # x2 : input
    lo = cc.trms .+ 1
    up = size(x2) .- cc.trms
    chcat(x2[UnitRange.(lo, up)...], x1)
end

utrim(l) = (192 - 2^(l + 3)) ÷ 2^l

"""
    unet(; inchannels)
Build a [U-Net](https://arxiv.org/abs/1505.04597v1) to process images with 
`inchannels` channels (`inchannels` = 3 for RGB images). This implementation 
corresponds to original paper with unpadded convolutions.
"""
function unet(; inchannels, volume = false) # 3D
    chd = 64
    chu = 2 * chd
    kc, kd, ku, kr = (3, 3), (2, 2), (2, 2), (1, 1) #outside
    if volume
        kc = (kc..., 3)

    end

    conv(chs) = Conv(kc, chs, relu)

    function conv2(chs)# vol, outside
        #k = vol ? (3, 3, 3) : (3, 3)
        #conv(chs) = Conv(k, chs, relu)
        ic, oc = chs
        conv(ic=>oc), conv(oc=>oc)
    end

    downsample() = MaxPool(kd) #k = vol ? (2, 2, 2) : (2, 2)

    function upsample(ic)#vol, outside
        #k = vol ? (2, 2, 2) : (2, 2)
        oc = ic ÷ 2
        ConvTranspose(k, ic=>oc, stride = k)
    end

    function encblock(chs, lvl)
        blk = []
        if lvl > 1
            push!(blk, downsample())
        end
        push!(blk, conv2(chs)...)
        blk
    end

    function decblock(chs, lvl)
        ic, oc = chs
        blk = []
        push!(blk, conv2(chs)...)
        if lvl == 1
            push!(blk, Conv(kr, oc=>1, sigmoid))
        else
            push!(blk, upsample(ic))
        end
        blk
    end

    enc, dec, con = [], [], []
    ice, oce = inchannels, 64 # number of input/ouput channels for encoder block
    icd, ocd = 2 * oce, oce   # number of input/ouput channels for decoder block
    for l ∈ 1:4
        push!(enc, encblock(ice=>oce, l))
        ice = oce
        oce *= 2

        push!(dec, decblock(icd=>ocd, l))
        icd *= 2
        ocd *= 2

        push!(con, CropCat(utrim(l), ndim)) # Center-crop concatenator
    end
    bdg = Chain(downsample(), conv2(oce=>icd)..., upsample(icd)) # bridge

    uchain(encoders = enc, decoders = dec, bridge = bdg, connection = con)

 #=   function uconv(ch)
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
        push!(con, CropCat((t, t, 0, 0)))
    end
    uchain(encoders = enc, decoders = dec, bridge = b, connection = con)
    =#
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