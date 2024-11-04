import numpy as np
from torchvision.models import VisionTransformer, resnet50
import timm


rng = np.random.default_rng()

model_dict = {
	'ResNet50': resnet50,
    'ViT-S/16': (lambda **kwargs: VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=1000,
        **kwargs,
    )),
    'ViT-B/16': (lambda **kwargs: VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=1000,
        **kwargs,
    )),
    'ViT-SS/16': (lambda **kwargs: VisionTransformer(
        image_size=112,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=1000,
        **kwargs,
    )),
    'tViT-S/16': (lambda **kwargs: timm.models.VisionTransformer(
        img_size=224,
        patch_size=16,
        depth=12,
        num_heads=6,
        embed_dim=384,
        num_classes=1000,
        **kwargs,
    )),
    'taViT-S/16': (lambda **kwargs: timm.models.VisionTransformer(
        img_size=224,
        patch_size=16,
        depth=12,
        num_heads=6,
        embed_dim=384,
        num_classes=1000,
        global_pool='avg',
        fc_norm=False,
        **kwargs,
    )),
    'traViT-S/16': (lambda **kwargs: timm.models.VisionTransformerRelPos(
        img_size=224,
        patch_size=16,
        depth=12,
        num_heads=6,
        embed_dim=384,
        num_classes=1000,
        weight_init='',
        **kwargs,
    )),
}
