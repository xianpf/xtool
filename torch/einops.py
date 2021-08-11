
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
scores = torch.einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale
x = torch.einsum('b h i j, b h j d -> b h i d', attention, values)
x = rearrange(x, 'b h n d -> b n (h d)')
Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_h, p2 = patch_w),
class_token = repeat(self.class_token, '() n d -> b n d', b = b)

