from torch import nn
from nnutils.layers import *


class TriplaneInfer3D(nn.Module):
    def __init__(self, num_layers, gf_dim, z_dim, norm):
        super().__init__()
        assert gf_dim % 3 == 0

        self.base = AdaBlock(None, gf_dim * 4, gf_dim * 4, z_dim, add_conv=False, relu='relu')

        dims_list = [gf_dim * 4 // 2**i for i in range(num_layers + 1)]
        self.out_dim = dims_list[-1]
        self.gblock2d_1 = build_gblock(2, num_layers, dims_list, k=3, d=2, p=1, op=1, adain=True,
                                       relu='leaky', z_dim=z_dim)
        std_dev = 0.02;
        w = 4
        self.const_w = nn.Parameter(torch.randn(1, gf_dim * 4, w, w) * std_dev)

        dims = [dims_list[-1] for i in range(int(math.log2(FLAGS.reso_vox // FLAGS.reso_3d)) + 1)]
        if len(dims) > 1:
            layers = build_gblock(3, 1, dims, False, norm=norm, relu='leaky', k=3, d=2, p=1, op=1, last_relu=False)
        else:
            layers = []
        layers.extend(build_gblock(3, 1, [dims[-1], 1], False, norm='none', relu='leaky', k=3, d=1, p=1, last_relu=False))
        self.voxel_net = nn.Sequential(
            *layers,
            nn.Sigmoid(),
        )

        range_idx = torch.arange(FLAGS.reso_3d)
        self.grid = torch.cartesian_prod(range_idx, range_idx, range_idx)

        self.out_dim = gf_dim 
        

    def forward(self, *input, **kwargs):
        """
        :param input: batch_z (N, Dz)
        :param kwargs:
        :return: some 3D representation: triplane (N, 3, C, W, H)
        """
        batch_z, _ = input

        _, C, H, W = self.const_w.size()
        N = batch_z.size(0)
        x = self.const_w.expand(N, C, H, W)
        x = self.base(batch_z, x)

        
        for gnet in self.gblock2d_1:
            x = gnet(batch_z, x)
        
        feat_triplane = x.view(N, 3, self.out_dim // 3, FLAGS.reso_3d, FLAGS.reso_3d)

        g = self.grid
        feat_world = torch.cat([
            feat_triplane[:, 0, :, g[:, 0], g[:, 1]],
            feat_triplane[:, 1, :, g[:, 0], g[:, 2]],
            feat_triplane[:, 2, :, g[:, 1], g[:, 2]]
        ], dim = 1).view(N, self.out_dim, FLAGS.reso_3d, FLAGS.reso_3d, FLAGS.reso_3d)

        feat_world = self.apply_symmetry(feat_world)

        vox_world = self.voxel_net(feat_world)
        return feat_world, vox_world

    def apply_symmetry(self, x):
        """
        apply reflection on W-axis
        :param x: (N, C, D, H, W)
        :return:
        """
        if FLAGS.apply_sym > 0:
            x_flip = torch.flip(x, dims=[-1])
            x = (x + x_flip) / 2
        return x
 