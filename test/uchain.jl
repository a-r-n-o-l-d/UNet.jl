@testset "uchain" begin
    struct Encoder
        dspl
        conv
    end

    @Flux.functor Encoder

    (e::Encoder)(x) = x |> e.dspl |> e.conv

    struct Decoder
        conv
        uspl
    end

    @Flux.functor Decoder

    (d::Decoder)(x) = x |> d.conv |> d.uspl



    enc1 = [Conv((3, 3), 1=>1, pad = 1) Conv((3, 3), 1=>1, pad = 1)]
    enc2 = Chain(MaxPool((2, 2)), Conv((3, 3), 1=>1, pad = 1))


    cv = Conv((3, 3), 4=>4)
    a = Chain(cv, cv)
    b = [cv cv]
    encs = [[a a] [b b]]
    for enc ∈ [a b], dec ∈ [a b], bdg ∈ [a b]
        #println("enc")
        #println(enc)
        uchain(encoders = enc, decoders = dec, bridge = bdg, connection = chcat)
    end
end