struct CenterCropCat
    trms
end

function CenterCropCat(t, vol)
    #t = utrim(lvl)
    t = vol ? (t, t, t) : (t, t)
    CenterCropCat((t..., 0, 0))
end

function (cc::CenterCropCat)(x1, x2) # x2 : input
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
function unet(; inchannels,
                nclasses = 1,
                volume = false,
                base = 64,
                batchnorm = false,
                padding = false,
                upsample = :convt,
                nlevels = 4)
    k₁ = volume ? (1, 1, 1) : (1, 1)
    k₂ = volume ? (2, 2, 2) : (2, 2)
    k₃ = volume ? (3, 3, 3) : (3, 3)
    
    # Return two convolutional layers
    function double_conv(c)
        ic, mc, oc = c
        p = padding ? 1 : 0
        cv1(f) = [Conv(k₃, f, pad = p, bias = false), BatchNorm(f[2], relu)]
        cv2(f) = [Conv(k₃, f, relu, pad = p)]
        cnv(f) = batchnorm ? cv1(f) : cv2(f)
        [cnv(ic => mc)..., cnv(mc => oc)...]
    end
    
    # Compute encoder/decoder number of channels for a given level (lvl)
    function level_channels(lvl)
        # encoder channels: input, middle, ouptput
        ice = (lvl == 1) ? inchannels : 2^(lvl - 2) * base
        mce = oce = 2^(lvl - 1) * base
        # decoder channels: input, middle, ouptput
        icd, mcd = 2 * oce, oce
        if upsample == :convt
            ocd = mcd
        else
            ocd = (lvl == 1) ? mcd : mcd ÷ 2
        end
        (ice, mce, oce), (icd, mcd, ocd)
    end

    # Downsampler, Upsampler
    function samplers(c)
        if upsample == :convt
            up =  ConvTranspose(k₂, c => (c ÷ 2), stride = 2)
        else
            up = Upsample(upsample, scale = k₂)
        end
        MaxPool(k₂), up
    end

    enc, dec, con = [], [], []
    for l ∈ 1:nlevels
        ec, dc = level_channels(l)
        if l == 1
            push!(enc, [double_conv(ec)...])
            if nclasses > 1
                d = volume ? 4 : 3
                cl = [Conv(k₁, dc[3] => nclasses), x -> softmax(x, dims = d)]
            else
                cl = [Conv(k₁, dc[3] => nclasses, sigmoid)]
            end
            push!(dec, [double_conv(dc)..., cl...])
        else
            dw, up = samplers(dc[3])
            push!(enc, [dw, double_conv(ec)...])
            push!(dec, [double_conv(dc)..., up])
        end
        if padding
            pusch!(con, chcat)
        else
            push!(con, CenterCropCat(utrim(l), volume))
        end
    end

    # bridge
    ec, dc = level_channels(nlevels + 1)
    ic, mc, _ = ec
    _, _, oc = dc
    dw, up = samplers(oc)
    bdg  = [dw, double_conv((ic, mc, oc))..., up]

    uchain(encoders = enc, decoders = dec, bridge = bdg, connection = con)
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