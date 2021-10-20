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
values correspond to the original paper implementation with unpadded
convolutions.

* `nclasses`: number of pixel classes. If `nclasses` = 1 the final layer have
  one channel with a sigmoid activation function, otherwise the final layer have
  `nclasses` channels followed by a softmax function.
* `volume`: set to `true` to process tri-dimensional datas
* `basewidth`: base number of channels, the number of filters/channels is
  multiplied by two at each U-Net level. Notice that `basewidth` should be a
  power of two number, other values could lead to unfunctionnal U-Net.
* `batchnorm`: if `true` add a `BatchNorm` layer after each convolution
* `padding`: if `true` convolutions are padded
* `upsample`: either `:convt` (`ConvTranspose` layer), `:nearest` or `:bilinear`
  (`Upsample layer`)
* `nlevels`: number of level or depth of the U-Net
"""
function unet(; inchannels,
                nclasses = 1,
                volume = false,
                basewidth = 64,
                # expansion = 2,
                batchnorm = false,
                padding = false,
                upsample = :convt,
                nlevels = 4)
    if volume && upsample == :bilinear
        throw(ArgumentError(
            "Bilinear upsampling can not be used with tri-dimensionnal datas."))
    end

    # Pack parameters in a named tuple for convenience
    pars = (k₁ = volume ? (1, 1, 1) : (1, 1), 
            k₂ = volume ? (2, 2, 2) : (2, 2),
            k₃ = volume ? (3, 3, 3) : (3, 3),
            nclasses = nclasses,
            inchannels = inchannels,
            basewidth = basewidth,
            batchnorm = batchnorm,
            padding = padding,
            upsample = upsample)

    # Build encoders and decoders for each level
    enc, dec, con = [], [], []
    for l ∈ 1:nlevels
        # e, d = level_enc_dec(l, pars)
        # push!(enc, e)
        # push!(dec, d)
        push!.((enc, dec), level_enc_dec(l, pars))
        if padding
            push!(con, chcat)
        else
            push!(con, CenterCropCat(trim(l, nlevels), volume))
        end
    end

    # Build bridge (i.e. bottom part)
    ec, dc = level_channels(nlevels + 1, pars)
    ic, mc, _ = ec
    _, _, oc = dc
    dw, up = samplers(oc, pars)
    bdg  = [dw, double_conv(ic, mc, oc, pars)..., up]

    uchain(encoders = enc, decoders = dec, bridge = bdg, connection = con)
end

# Build encoder and decoder blocks for a given level
function level_enc_dec(lvl, pars)
    # number of channels (input, middle, output)
    ec, dc = level_channels(lvl, pars)
    enc = []
    dec = []
    if lvl == 1
        enc = double_conv(ec, pars)
        # build classifier layers
        k₁ = pars[:k₁]
        nc = pars[:nclasses]
        if nc > 1
            d = length(k₁)
            cl = [Conv(k₁, dc[3] => nc), x -> softmax(x, dims = d)]
        else
            cl = [Conv(k₁, dc[3] => 1, sigmoid)]
        end
        dec = [double_conv(dc, pars)..., cl...]
    else
        dw, up = samplers(dc[3], pars)
        enc = [dw, double_conv(ec, pars)...]
        dec = [double_conv(dc, pars)..., up]
    end
    enc, dec
end

# Return two convolutional layers, with batchnom if needed
function double_conv(chs, pars)
    ic, mc, oc = chs
    padding, k₃, batchnorm = pars[:padding], pars[:k₃], pars[:batchnorm]
    p = padding ? 1 : 0
    cv1(chs) = [Conv(k₃, chs, pad = p, bias = false), BatchNorm(chs[2], relu)]
    cv2(chs) = [Conv(k₃, chs, relu, pad = p)]
    cnv(chs) =  batchnorm ? cv1(chs) : cv2(chs)
    [cnv(ic => mc)..., cnv(mc => oc)...]
end

double_conv(ic, mc, oc, pars) = double_conv((ic, mc, oc), pars)

# Compute encoder/decoder number of channels at a given level (lvl)
# return two tuples one for encoder and one for decoder
# formula : 
#  ice = expansion^(lvl - 2) * basewidth
#  mce = expansion^(lvl - 1) * basewidth
function level_channels(lvl, pars)
    inchannels, basewidth = pars[:inchannels], pars[:basewidth]
    upsample = pars[:upsample]
    # encoder channels: input, middle, ouptput = (ice, mce, oce)
    ice = (lvl == 1) ? inchannels : 2^(lvl - 2) * basewidth
    mce = oce = 2^(lvl - 1) * basewidth
    # decoder channels: input, middle, ouptput = (icd, mcd, ocd)
    icd, mcd = 2 * oce, oce
    if upsample == :convt
        ocd = mcd
    else
        ocd = (lvl == 1) ? mcd : mcd ÷ 2
    end
    (ice, mce, oce), (icd, mcd, ocd)
end

# Downsampler, Upsampler
function samplers(chs, pars)
    k₂, upsample = pars[:k₂], pars[:upsample]
    dw = MaxPool(k₂)
    if upsample == :convt
        up =  ConvTranspose(k₂, chs => (chs ÷ 2), stride = 2)
    else
        up = Upsample(upsample, scale = k₂)
    end
    dw, up
end