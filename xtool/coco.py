

from maskrcnn_benchmark.data import datasets as D
from maskrcnn_benchmark.data.datasets.coco import COCODataset
# dataset = D.ConcatDataset([COCODataset(root='datasets/coco/train2014', ann_file='datasets/coco/annotations/instances_train2014.json', remove_images_without_annotations=True), COCODataset(root='datasets/coco/val2014', ann_file='datasets/coco/annotations/instances_valminusminival2014.json', remove_images_without_annotations=True)])
dataset = COCODataset(root='datasets/coco/train2014', ann_file='datasets/coco/annotations/instances_train2014.json', remove_images_without_annotations=True)




def render_mask(img, masks):
    import pdb; pdb.set_trace()

labels = dataset[0][1].extra_fields['labels'].numpy()
masks = dataset[0][1].extra_fields['masks']
import pdb; pdb.set_trace()