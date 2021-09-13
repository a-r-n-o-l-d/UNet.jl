var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = UNet","category":"page"},{"location":"#UNet","page":"Home","title":"UNet","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for UNet.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [UNet]","category":"page"},{"location":"#UNet.chcat-Tuple","page":"Home","title":"UNet.chcat","text":"chcat(x...)\n\nConcatenate the image data along the dimension corresponding to the channels. Image data should be stored in WHCN order (width, height, channels, batch) or WHDCN (width, height, depth, channels, batch) in 3D context. Channels are assumed to be the penultimate dimension.\n\nExample\n\njulia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels\n\njulia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels\n\njulia> chcat(x1, x2) |> size\n(32, 32, 8, 6)\n\n\n\n\n\n","category":"method"},{"location":"#UNet.checksize-Tuple{Int64}","page":"Home","title":"UNet.checksize","text":"checksize(sz; padding, nlevels)\n\nCheck if image size is appropriate with a given U-Net architecture. sz could be  an Int or a Tuple of Int.\n\n\n\n\n\n","category":"method"},{"location":"#UNet.uchain-Tuple{}","page":"Home","title":"UNet.uchain","text":"uchain(;encoders, decoders, bridge, connection)\n\nBuild a Chain with U-Net like architecture. encoders and decoders are  arrays of encoding/decoding blocks, from top to bottom (see diagram below).  bridge is the bottom part of U-Net  architecture. Each level of the U-Net is connected through a 2-argument callable connection. connection could be an array in case the way levels are connected vary from  one level to another.\n\nNotes :\n\nusually encoder block starts with a 'MaxPool' to downsample image by 2, except\n\nthe first level encoder.\n\nusually decoder block ends with a 'ConvTranspose' to upsample image by 2,\n\nexcept the first level decoder.\n\n+---------+                                                          +---------+\n|encoder 1|                                                          |decoder 1|\n+---------+                                                          +---------+\n     |------------------------------------------------------------------->^     \n     |                                                                    |     \n     |   +---------+                                         +---------+  |     \n     +-->|encoder 2|                                         |decoder 2|--+     \n         +---------+                                         +---------+        \n              |-------------------------------------------------->^             \n              |                                                   |             \n              |   +---------+                        +---------+  |             \n              +-->|encoder 3|                        |decoder 3|--+             \n                  +---------+                        +---------+                \n                       |--------------------------------->^                     \n                       |                                  |                     \n                       |   +---------+       +---------+  |                     \n                       +-->|encoder 4|       |decoder 4|--+                     \n                           +---------+       +---------+                        \n                                |---------------->^                             \n                                |                 |                             \n                                |    +--------+   |                             \n                                +--->| bridge |---+                             \n                                     +--------+                                  \n\nSee also chcat.\n\n\n\n\n\n","category":"method"},{"location":"#UNet.unet-Tuple{}","page":"Home","title":"UNet.unet","text":"unet(; inchannels, nclasses = 1, volume = false, base = 64,\nbatchnorm = false, padding = false, upsample = :convt, nlevels = 4)\n\nBuild a U-Net to process images with  inchannels channels (e.g. inchannels = 3 for RGB images). Default argument values correspond to the original paper with unpadded convolutions.\n\nnclasses: number of pixel classes\nvolume: set to true to process tri-dimensional datas\nbase: base number of convolution filters, the number of filters/channels is\n\nmultiplied by two at each U-Net level\n\nbatchnorm: if true add a BatchNorm layer after each convolution\npadding: if true convolutions are padded\nupsample: either :convt (ConvTranspose layer), :nearest or :bilinear\n\n(Upsample layer)\n\nnlevels: number of level or depth of the U-Net\n\n\n\n\n\n","category":"method"}]
}
