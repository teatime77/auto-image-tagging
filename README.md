# auto-img-tag
automatic image tagging by chroma key


```bash
odtk train model.pth --images /content/work/img --annotations /content/work/train.json  --backbone ResNet50FPN --lr 0.00005 --fine-tune /content/work/retinanet_rn50fpn.pth --val-images /content/work/img --val-annotations /content/work/train.json --classes 4 --jitter 688 848 --resize 768 --augment-rotate --augment-brightness 0.01 --augment-contrast 0.01 --augment-hue 0.002 --augment-saturation 0.01 --batch 4 --regularization-l2 0.0001 --iters 15000 --val-iters 1000 --rotated-bbox

```bash
odtk infer model.pth --images=/content/work/img --output=detections.json --batch 8
```