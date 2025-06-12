# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import matplotlib.image

import matplotlib.pyplot as plt
import segment_this_thing
import torch

from segment_this_thing import Foveator, SegmentThisThingPredictor


def main() -> None:
    arg_parser = argparse.ArgumentParser("Foveation Test")
    arg_parser.add_argument("--input", required=True)
    arg_parser.add_argument("--weights", required=True)
    arg_parser.add_argument("--model-size", choices=["b", "h", "l"], default="l")
    arg_parser.add_argument("--show-all", action="store_true")
    arg_parser.add_argument("--cpu", action="store_true")
    args = arg_parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not args.cpu
        else torch.device("cpu")
    )

    token_size = 16

    foveation_pattern = Foveator(
        token_size=token_size, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
    ).to(device)

    model_builder = getattr(
        segment_this_thing, f"build_segment_this_thing_{args.model_size}"
    )
    model = model_builder(
        num_tokens=foveation_pattern.get_num_tokens(),
        token_size=token_size,
    )
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    model = model.to(device)
    print(model)

    predictor = SegmentThisThingPredictor(model, foveation_pattern)

    image = torch.from_numpy(matplotlib.image.imread(args.input))

    fig, axs = plt.subplots(1, 1)
    axs = [axs]
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    for ax in axs:
        ax.set_axis_off()

    down = None

    def onpress(event):
        nonlocal down

        down = torch.tensor([int(event.xdata), int(event.ydata)])

    def onrelease(event):
        nonlocal fig
        nonlocal axs
        nonlocal down

        if event.inaxes != axs[0] or event.button != 1 or event.dblclick:
            return

        center = torch.tensor([int(event.xdata), int(event.ydata)])

        if torch.linalg.norm((center - down).float()) > 10.0:
            # ignore if the mouse moved too much
            return

        with torch.no_grad():
            masks, ious, foveation = predictor.get_prediction(
                image.to(device), center.to(device), return_foveation=True
            )

        if args.show_all:
            ious, inds = torch.sort(ious, descending=True)

            masks = masks[inds]

        else:
            k = ious.argmax().item()
            ious = ious[k : k + 1]
            masks = masks[k : k + 1]

        num_masks = len(masks)
        num_plots = 1 + num_masks

        if len(axs) != num_plots:
            # reconfigure the plot to add additional images as needed
            for ax in axs:
                fig.delaxes(ax)
            axs = [fig.add_subplot(1, num_plots, i + 1) for i in range(num_plots)]
            for ax in axs:
                ax.set_axis_off()

            axs[0].imshow(image)
            axs[0].set_title("Input Image")
            plt.tight_layout()

        for k, (mask, iou) in enumerate(zip(masks, ious)):
            segmentation = foveation_pattern.generate_foveated_visualization(
                mask.unsqueeze(1)
            ).sigmoid()

            recon = foveation_pattern.generate_foveated_visualization(foveation)

            axs[1 + k].imshow(
                torch.where(
                    segmentation > 0.5,
                    (
                        0.5 * recon.float()
                        + 0.5 * torch.tensor([0x32, 0xA8, 0x52]).view(3, 1, 1)
                    ),
                    recon,
                )
                .permute(1, 2, 0)
                .byte()
                .cpu()
                .numpy(),
            )
            axs[1 + k].set_title(f"IoU = {iou.item():0.3f}")

        plt.draw()

    fig.canvas.mpl_connect("button_press_event", onpress)
    fig.canvas.mpl_connect("button_release_event", onrelease)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
