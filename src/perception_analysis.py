import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    evals = np.load('perception_eval.npz')
    grasp_evals = np.load('grasp_eval.npz')

    simple_eval = evals['simple']
    cosine_eval = evals['cosine']
    bayes_eval = evals['bayes']
    sam3_eval = evals['sam3']
    print(f'simple: {simple_eval}')
    print(f'cosine: {cosine_eval}')
    print(f'bayes: {bayes_eval}')
    print(f'sam3: {sam3_eval}')
    simple_grasp_eval = grasp_evals['grasp_simple']
    cosine_grasp_eval = grasp_evals['grasp_cosine']
    bayes_grasp_eval = grasp_evals['grasp_bayes']
    sam3_grasp_eval = grasp_evals['grasp_sam3']
    print(f'simple grasp: {simple_grasp_eval}')
    print(f'cosine grasp: {cosine_grasp_eval}')
    print(f'bayes grasp: {bayes_grasp_eval}')
    print(f'sam3 grasp: {sam3_grasp_eval}')

    simple_segments_total = np.sum(simple_eval, axis=0)
    cosine_segments_total = np.sum(cosine_eval, axis=0)
    bayes_segments_total = np.sum(bayes_eval, axis=0)
    sam3_segments_total = np.sum(sam3_eval, axis=0)

    simple_grasps_total = np.sum(simple_grasp_eval, axis=0)
    cosine_grasps_total = np.sum(cosine_grasp_eval, axis=0)
    bayes_grasps_total = np.sum(bayes_grasp_eval, axis=0)
    sam3_grasps_total = np.sum(sam3_grasp_eval, axis=0)

    task_ids = [i+1 for i in range(6)]

    objects = [f'Object {task_num}' for task_num in task_ids]
    task_nums = {
        'Successful Segmentation': simple_segments_total,
        'Successful Grasp': simple_grasps_total,
    }

    x = np.arange(len(objects))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in task_nums.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of successes')
    ax.set_title('Successful segmentations & grasps for \n SAM-CLIP point cloud concatenation')
    ax.set_xticks(x + width, objects)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 20)

    plt.savefig("plots/simple.png")

    objects = [f'Object {task_num}' for task_num in task_ids]
    task_nums = {
        'Successful Segmentation': cosine_segments_total,
        'Successful Grasp': cosine_grasps_total,
    }

    x = np.arange(len(objects))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in task_nums.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of successes')
    ax.set_title('Successful segmentations & grasps for \n SAM-CLIP cosine similarity averaging over multiple views')
    ax.set_xticks(x + width, objects)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 20)

    plt.savefig("plots/cosine.png")

    objects = [f'Object {task_num}' for task_num in task_ids]
    task_nums = {
        'Successful Segmentation': bayes_segments_total,
        'Successful Grasp': bayes_grasps_total,
    }

    x = np.arange(len(objects))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in task_nums.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of successes')
    ax.set_title('Successful segmentations & grasps for \n SAM-CLIP Bayesian Inference over multiple views')
    ax.set_xticks(x + width, objects)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 20)

    plt.savefig("plots/bayes.png")

    objects = [f'Object {task_num}' for task_num in task_ids]
    task_nums = {
        'Successful Segmentation': sam3_segments_total,
        'Successful Grasp': sam3_grasps_total,
    }

    x = np.arange(len(objects))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in task_nums.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of successes')
    ax.set_title('Successful segmentations & grasps for \n SAM3 point cloud concatenation')
    ax.set_xticks(x + width, objects)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 20)

    plt.savefig("plots/sam3.png")

if __name__ == "__main__":
    main()