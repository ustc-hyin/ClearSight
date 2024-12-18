python ./visaug/analysis/vis_flow.py \
    --model-path /model/llava \
    --question-file ./data/pope/coco/coco_pope_random.json \
    --image-folder ./data/pope/coco/val2014 \
    --answers-file ./outputs/analysis/res_coco_random.pt 

python ./visaug/analysis/analysis_plot.py