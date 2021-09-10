# Encoder structure : a downsampling is applied then two convolutions are done
struct Encoder
    dspl
    conv
end

@Flux.functor Encoder

(e::Encoder)(x) = x |> e.dspl |> e.conv

# Decoder structure : two convolutions are done then an upsampling is applied
struct Decoder
    conv
    uspl
end

@Flux.functor Decoder

(d::Decoder)(x) = x |> d.conv |> d.uspl

# Bridge structure :
#   - downsampling
#   - two convolutions
#   - upsampling
struct Bridge
    dspl
    conv
    uspl
end

@Flux.functor Bridge

(b::Bridge)(x) = x |> b.dspl |> b.conv |> b.uspl

# Encoder builders
function encoder(lvl)
    enc = []
    if lvl == 1
        push!(enc, identity)
    else
        push!(enc, MaxPool((2, 2)))
    end
    push!(enc, Conv((3, 3), 1=>1, pad = 1))
    push!(enc, Conv((3, 3), 1=>1, pad = 1))
    enc
end

encoder(t::Symbol, lvl) = encoder(Val(t), lvl)

encoder(t::Val{:array}, lvl) = encoder(lvl)

encoder(t::Val{:chain}, lvl) = Chain(encoder(lvl)...)

encoder(t::Val{:struct}, lvl) = begin
    enc = encoder(lvl)
    Encoder(enc[1], Chain(enc[2], enc[3]))
end

# Decoder builders
function decoder(lvl)
    dec = Any[Conv((3, 3), 2=>1, pad = 1), Conv((3, 3), 1=>1, pad = 1)]
    if lvl == 1
        push!(dec, identity)
    else
        push!(dec, Upsample(:bilinear, scale = 2))
    end
    dec
end

decoder(t::Symbol, lvl) = decoder(Val(t), lvl)

decoder(t::Val{:array}, lvl) = decoder(lvl)

decoder(t::Val{:chain}, lvl) = Chain(decoder(lvl)...)

decoder(t::Val{:struct}, lvl) = begin
    dec = decoder(lvl)
    Decoder(Chain(dec[1], dec[2]), dec[3])
end

# Bridge builders
bridge() = [MaxPool((2, 2)),
            Conv((3, 3), 1=>1, pad = 1), Conv((3, 3), 1=>1, pad = 1),
            Upsample(:bilinear, scale = 2)]

bridge(t::Symbol) = bridge(Val(t))

bridge(t::Val{:array}) = bridge()

bridge(t::Val{:chain}) = Chain(bridge()...)

bridge(t::Val{:struct}) = begin
    bdg = bridge()
    Bridge(bdg[1], Chain(bdg[2], bdg[3]), bdg[4])
end

@testset "uchain" begin
    blkt = [:array :chain :struct]
    for enct ∈ blkt, dect ∈ blkt, bdgt ∈ blkt
        enc, dec = [], []
        for l ∈ 1:2
            push!(enc, encoder(enct, l))
            push!(dec, decoder(dect, l))
        end
        bdg = bridge(bdgt)
        model = uchain(encoders = enc, decoders = dec, bridge = bdg, connection = chcat)
        x = rand(Float32, 32, 32, 1, 1)
        @test (model(x) |> size) == (32, 32, 1, 1)
    end
end