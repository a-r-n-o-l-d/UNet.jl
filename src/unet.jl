struct CenterCropCat
    trms
end

function CenterCropCat(t, vol)
    t = vol ? (t, t, t) : (t, t)
    CenterCropCat((t..., 0, 0))
end

function (cc::CenterCropCat)(x1, x2) # x2 : input
    lo = cc.trms .+ 1
    up = size(x2) .- cc.trms
    chcat(x2[UnitRange.(lo, up)...], x1)
end

"""
    unet(; inchannels, nclasses = 1, volume = false, base = 64,
    batchnorm = false, padding = false, upsample = :convt, nlevels = 4)
Build a [U-Net](https://arxiv.org/abs/1505.04597v1) to process images with 
`inchannels` channels (e.g. `inchannels` = 3 for RGB images). Default argument
values corresponds to original paper with unpadded convolutions.

- nclasses: number of pixel classes
- volume: set to `true` to process tri-dimentional datas
- base: base number of convolution filters, the number of filters/channels is
multiplied by two at each U-Net level
- batchnorm: if `true` add a `BatchNorm` layer after each convolution
- padding: if `true` convolutions are padded
- upsample: either `:convt` (`ConvTranspose` layer), `:nearest` or `:bilinear`
(`Upsample layer`)
- nlevels: number of level or depth of the U-Net
"""
function unet(; inchannels,
                nclasses = 1,
                volume = false,
                base = 64,
                batchnorm = false,
                padding = false,
                upsample = :convt,
                nlevels = 4)
    if volume && upsample == :bilinear
        throw(ArgumentError(
            "Bilinear upsampling can not be used with tri-dimensionnal datas."))
    end
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
            push!(con, chcat)
        else
            push!(con, CenterCropCat(trim(l, nlevels), volume))
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