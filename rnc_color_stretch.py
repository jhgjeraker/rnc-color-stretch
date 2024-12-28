# Adapted from Roger N. Clark's `rnc-color-stretch` version 0.975
# by Johannes H. Gjeraker.
# https://github.com/jhgjeraker
#
#  Copyright (c) 2016, Roger N. Clark, clarkvision.com
#
# http://www.clarkvision.com/articles/astrophotography.software/rnc-color-stretch/
#
# All rights reserved.
#
# GNU General Public License https://www.gnu.org/licenses/gpl.html
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  - Redistributions of the program must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  - Neither Roger N. Clark, clarkvision.com nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Notations on the form `Line #` indicated the starting position
# of the related logic in the original davinci implementation.

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_sysargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
    )

    parser.add_argument(
        "image",
        type=str,
        help="path to target image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_output",
        metavar="",
        help="directory into which files are written",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="visualize each step in order",
    )
    parser.add_argument(
        "--write-plot",
        action="store_true",
        help="write each visualization to file",
    )
    parser.add_argument(
        "--tone-curve",
        action="store_true",
        help="apply a tone curve to the image",
    )
    parser.add_argument(
        "--s-curve",
        type=int,
        default=0,
        help="s-curve application, "
        "0: no s-curve application, "
        "1: apply a s-curve stretch, "
        "2: apply a stronger s-curve stretch, "
        "3: apply s-curve 1 then 2, "
        "4: apply s-curve 2 then 1",
    )
    parser.add_argument(
        "--skylevelfactor",
        type=float,
        default=0.06,
        metavar="",
        help="sky level relative to the histogram peak",
    )
    parser.add_argument(
        "--zerosky-red",
        type=int,
        default=4096,
        metavar="",
        help="desired zero point on sky, red channel",
    )
    parser.add_argument(
        "--zerosky-green",
        type=int,
        default=4096,
        metavar="",
        help="desired zero point on sky, green channel",
    )
    parser.add_argument(
        "--zerosky-blue",
        type=int,
        default=4096,
        metavar="",
        help="desired zero point on sky, blue channel",
    )
    parser.add_argument(
        "--rootpower",
        type=int,
        default=6,
        metavar="",
        help="power factor 1/rootpower",
    )
    parser.add_argument(
        "--rootpower2",
        type=int,
        default=1,
        metavar="",
        help="user if rootiter == 2 and rootpower2 > 1",
    )
    parser.add_argument(
        "--rootiter",
        type=int,
        default=1,
        metavar="",
        help="iterations for applying rootpower - sky",
    )
    parser.add_argument(
        "--setmin",
        action="store_true",
        help="modification for minimum",
    )
    parser.add_argument(
        "--setmin-red",
        type=int,
        default=0,
        metavar="",
        help="minimum for red",
    )
    parser.add_argument(
        "--setmin-green",
        type=int,
        default=0,
        metavar="",
        help="minimum for green",
    )
    parser.add_argument(
        "--setmin-blue",
        type=int,
        default=0,
        metavar="",
        help="minimum for blue",
    )
    parser.add_argument(
        "--no-colorcorrection",
        action="store_true",
        help="disable color correction to output image",
    )
    parser.add_argument(
        "--colorenhance",
        type=float,
        default=1.0,
        metavar="",
        help="color enhancement value",
    )

    args = parser.parse_args()

    print("- All parameters:")
    print(f"  Input file:       {args.image}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"  Plot:             {args.plot}")
    print(f"  Write Plot:       {args.write_plot}")
    print(f"  Tone Curve:       {args.tone_curve}")
    print(f"  S-Curve:          {args.s_curve}")
    print(f"  Skylevelfactor:   {args.skylevelfactor}")
    print(f"  Zerosky Red:      {args.zerosky_red}")
    print(f"  Zerosky Green:    {args.zerosky_green}")
    print(f"  Zerosky Blue:     {args.zerosky_blue}")
    print(f"  Rootpower:        {args.rootpower}")
    print(f"  Rootpower2:       {args.rootpower2}")
    print(f"  Rootiter:         {args.rootiter}")
    print(f"  Setmin:           {args.setmin}")
    print(f"  Setmin Red:       {args.setmin_red}")
    print(f"  Setmin Green:     {args.setmin_green}")
    print(f"  Setmin Blue:      {args.setmin_blue}")
    print(f"  Colorcorrect:     {not args.no_colorcorrection}")
    print(f"  Colorenchance:    {args.colorenhance}")

    return args


def histogram(img: np.ndarray, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the histogram for a single channel.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape [x, y, 3].
    channel : int
        Channel for which the histogram is calculated.

    Returns
    -------
    histogram : np.ndarray
        Values for each bin in the created histogram.
    bins : np.ndarray
        Histogram bins in the specified range.

    """

    return np.histogram(img[:, :, channel], range=(0, 65535), bins=65536)


def imshow(
    in_img: np.ndarray,
    name: str,
    args: argparse.Namespace,
    flip: bool = True,
) -> None:
    """
    Helper function for plotting/writing an image + channel histograms.
    Will plot and/or write to file depending on user configuration.

    Parameters
    ----------
    in_img : np.ndarray
        Image that is to be plotted/written.
    name : str
        Image name used in writefile and/or plot title.
    args : argparse.Namespace
        User-provided system arguments.
    flip : bool
        Whether or not to flip image axes.
        Defaults to True.

    """

    if not args.plot and not args.write_plot:
        return

    # Swap x- and y- axis back to original state to
    # compensate for flipping them at initial read.
    if flip:
        img = np.swapaxes(in_img, 0, 1)
    else:
        img = in_img

    _, ax = plt.subplots(1, 2, figsize=(24, 8))
    ax[0].imshow((img - np.min(img)) / (np.max(img) - np.min(img)))

    for ch, color in enumerate(["red", "green", "blue"]):
        hist, bins = histogram(img, ch)
        ax[1].plot(bins[:-1], hist, color=color)

    plt.title(name)
    plt.tight_layout()

    if args.plot:
        plt.show()
    if args.write_plot:
        plt.savefig(f"{args.output_dir}/{name}.jpg")


def print_ch_moments(img: np.ndarray, header: str, indent: int = 0) -> None:
    """
    Print image minimum, maximum, and average value per channel.

    Parameters
    ----------
    img : np.array
        Image for which moments are printed.
    header : str
        Output prefix, usually name of current step.
    indent : int
        Number of spaces to indent the output.

    """

    print(f'{indent*" "}- {header}\n')
    print(f'{indent*" "}  Channel       Min        Max       Mean')
    print(f'{indent*" "}  ---------------------------------------')
    for ch, color in enumerate(["Red", "Green", "Blue"]):
        print(
            "{}  {:6s}  {:9.2f}  {:9.2f}  {:9.2f}".format(
                indent * " ",
                color,
                np.min(img[:, :, ch]),
                np.max(img[:, :, ch]),
                np.mean(img[:, :, ch]),
            )
        )
    print()


def env_setup(args: argparse.Namespace) -> None:
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


def read_img(path: str) -> np.ndarray:
    """
    Read and pre-process image from provided path.

    Parameters
    ----------
    path : str
        Target image path.

    Returns
    -------
    img : np.ndarray
        Read and processed image array of shape [x, y, 3].

    """

    # Line 357.
    #
    # Parameter `cv2.IMREAD_UNCHANGED` preserves 16-bit depth.
    img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Sanity check for number of channels.
    # Only accept 3 channels.
    if img_bgr.shape[2] != 3:
        print("- ERROR")
        print("  Input image is not a 3 channel image.")
        print("  Number of channels found: {}".format(img_bgr.shape[2]))
        print("  Exiting...")
        sys.exit()

    # Opencv imread() return the color channels in reverse order.
    # Blue  Channel: img[:, :, 0]
    # Green Channel: img[:, :, 1]
    # Red   Channel: img[:, :, 2]
    # These is reversed to RGB in order to maintain sanity.
    img = np.zeros(img_bgr.shape).astype(img_bgr.dtype)
    img[:, :, 0] = img_bgr[:, :, 2]
    img[:, :, 1] = img_bgr[:, :, 1]
    img[:, :, 2] = img_bgr[:, :, 0]

    print(
        "- input image dimensions: {:d} {:d} {:d}".format(
            img.shape[0],
            img.shape[1],
            img.shape[2],
        )
    )

    # Images are usually represented with the x-axis from left-to-right
    # and with the y-axis from top-to-bottom.
    #
    # Numpy ndarrays are represented as axis 0 for rows, and
    # axis 1 for rows.
    #
    # In order ot use the axis notation of [x, y, z], flip the
    # image before any further processing is done.
    # We must, of course, remember to flip it back before output.
    img = np.swapaxes(img, 0, 1)

    print(f"\n- Image of type {img.dtype}:")
    print(f"  - Min:  {np.min(img):8.2f}")
    print(f"  - Max:  {np.max(img):8.2f}")
    print(f"  - Mean: {np.mean(img):8.2f}")

    return img


def format_and_scale(img: np.ndarray) -> np.ndarray:
    """
    Depending on the input image parameters, scale and/or change the
    image format to achieve proper base values for further processing.

    If a permutation of filetype, range, or scale is not supported, this
    is likely where we need to add more configurations.

    Parameters
    img : np.ndarray
        Image of shape [x, y, 3] to be scaled.
    img : np.ndarray
        Scaled and/or formatted image array of shape [x, y, 3].

    """

    # Line 406.
    if np.max(img) < 1.00001 and np.issubdtype(img.dtype, np.floating):
        print("- Scaling float data by 65535 to 16-bit range.")
        img = img * 65535

    elif np.max(img) >= 1.00001 and np.issubdtype(img.dtype, np.floating):
        print("- Scaling float data so that max is 65000.")
        img = img * (65000 / np.max(img))

    elif np.max(img) > 8000 and np.issubdtype(img.dtype, np.integer):
        print("- Image integers have good data range.")

    elif img.dtype == np.int16 and np.max(img) > 16000 and np.max(img) < 32768:
        ascale = 65000 / np.max(img)
        img = img * ascale
        print("- Image integers are signed 16-bit range.")
        print(f"  Scaling by factor {ascale:.2f}.")
        if np.min(img) < 0:
            print("  Negative pixels will be truncated to 0.")
            print(f"  Number of negative pixels found: {len(img[img < 0])}")
            img[img < 0] = 0

    elif img.dtype == np.int16 and np.max(img) < 16000:
        print("- Image integers are signed 16-bit range.")
        print(f"  Maximum value {np.max(img):.2f} < 16000 seems too low.")
        print("  Exiting...")

    elif np.max(img) <= 8000 and np.dtype in [
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
    ]:
        print(f"- Integer max value {np.max(img)} should be > 8000.")
        print("   Exiting.")

    else:
        print(f"- Unimplemented data type {img.dtype}.")
        print("  Convert the image to 16-bit tif.")
        print("  Exiting...")

    print("")
    return img


def tone_curve(img: np.ndarray) -> np.ndarray:
    """
    Apply a tone curve to the input image.
    This can be useful for very dark images, but might lead
    to overexposure if used where not necessary.

    Parameters
    ----------
    img : np.ndarray
        Image onto which the curve is applied.

    Returns
    -------
    img : np.ndarray
        Image after applying the tone curve.

    """

    print("- Applying tone curve to input image.")
    b = 12.0
    c = 65535.0
    d = 12.0
    img_tone_curve = img * b * ((1.0 / d) ** ((img / c) ** (0.4)))
    print_ch_moments(
        img=img_tone_curve,
        header="Input image after application of tone curve.",
    )

    return img_tone_curve


def smooth_and_subtract(
    in_img: np.ndarray,
    skylevelfactor: float,
    zerosky_r: int,
    zerosky_g: int,
    zerosky_b: int,
    indent_level: int = 0,
) -> np.ndarray:
    # Smooth the histogram so we can find the darkest sky level
    # to subtract and find the sky histogram peak, which should be
    # close to the darkest deep space zero level.
    #
    # Make a copy of the input image.
    img = np.ones(in_img.shape) * in_img

    # The function `smooth_and_subtract` might be called insider other
    # functions where it would be appropriate to indent all output.
    ni = "  " * indent_level

    print(f"{ni}- Computing smoothed RGB histograms on image.")
    # Do two passes on finding sky level.
    for i in range(2):
        print(f"{ni}  - Pass {i+1}")

        hist_r, _ = histogram(img, 0)
        hist_g, _ = histogram(img, 1)
        hist_b, _ = histogram(img, 2)

        # Arrays for smoothed histograms.
        hist_r_sm = np.ones(len(hist_r)) * hist_r
        hist_g_sm = np.ones(len(hist_g)) * hist_g
        hist_b_sm = np.ones(len(hist_b)) * hist_b

        # Smoothing window width.
        ism = 300

        # Apply histogram smoothing.
        for j in range(65536):
            j_lo = max([j - ism, 0])
            j_hi = min([j + ism + 1, 65536])

            hist_r_sm[j] = np.mean(hist_r[j_lo:j_hi])
            hist_g_sm[j] = np.mean(hist_g[j_lo:j_hi])
            hist_b_sm[j] = np.mean(hist_b[j_lo:j_hi])

        # Now find the maximum values.
        # Limit the range in case of clipping or saturation.
        hist_r_sm_argmax = np.argmax(hist_r_sm[400 : 65500 + 1]) + 400
        hist_g_sm_argmax = np.argmax(hist_g_sm[400 : 65500 + 1]) + 400
        hist_b_sm_argmax = np.argmax(hist_b_sm[400 : 65500 + 1]) + 400

        print(f"{ni}    - Histogram peak.\n")
        print(f"{ni}      Channel  Index      Value")
        print(f"{ni}      -------------------------")
        print(
            "{}          Red  {:5d}  {:9.2f}".format(
                ni,
                hist_r_sm_argmax,
                hist_r_sm[hist_r_sm_argmax],
            )
        )
        print(
            "{}        Green  {:5d}  {:9.2f}".format(
                ni,
                hist_g_sm_argmax,
                hist_g_sm[hist_g_sm_argmax],
            )
        )
        print(
            "{}         Blue  {:5d}  {:9.2f}".format(
                ni,
                hist_b_sm_argmax,
                hist_b_sm[hist_b_sm_argmax],
            )
        )
        print("")

        # Line 807
        # Now find the sky level on the left side of the histogram.
        hist_r_sky = hist_r_sm[hist_r_sm_argmax] * skylevelfactor
        hist_g_sky = hist_g_sm[hist_g_sm_argmax] * skylevelfactor
        hist_b_sky = hist_b_sm[hist_b_sm_argmax] * skylevelfactor

        hist_r_sky_index = 0
        hist_g_sky_index = 0
        hist_b_sky_index = 0

        # Search from max towards left minimum, but search
        # for the green level in each color.
        for j in range(hist_r_sm_argmax, 0, -1):
            if (
                hist_r_sm[j] >= hist_g_sky
                and hist_r_sm[j - 1] <= hist_g_sky
                and hist_r_sky_index == 0
            ):
                hist_r_sky_index = j
                break

        for j in range(hist_g_sm_argmax, 0, -1):
            if (
                hist_g_sm[j] >= hist_g_sky
                and hist_g_sm[j - 1] <= hist_g_sky
                and hist_g_sky_index == 0
            ):
                hist_g_sky_index = j
                break

        for j in range(hist_b_sm_argmax, 0, -1):
            if (
                hist_b_sm[j] >= hist_g_sky
                and hist_b_sm[j - 1] <= hist_g_sky
                and hist_b_sky_index == 0
            ):
                hist_b_sky_index = j
                break

        # Line 843
        if (
            hist_r_sky_index == 0
            or hist_g_sky_index == 0
            or hist_b_sky_index == 0
        ):
            print(f"{ni}- Histogram sky level {skylevelfactor:.2f} not found.")
            print(
                "{}  Channels: Red={}, blue={}, green={}".format(
                    ni,
                    hist_r_sky_index,
                    hist_g_sky_index,
                    hist_b_sky_index,
                )
            )
            print("")
            print(f"{ni}  Image is likely too dark. but you can try again")
            print(f"{ni}  Suggestions:")
            print(f"{ni}    - Add a --tone-curve.")
            print(f"{ni}    - Reduce --s-curve intensity.")
            print(f"{ni}    - Fewer --rootiter or smaller --rootpower2.")
            print(f"{ni}  Exiting...")
            sys.exit()

        # Line 882
        print(
            f"{ni}    - Histogram dark sky level, "
            f"{skylevelfactor*100:.2f}% of max.\n"
        )
        print(f"{ni}      Channel  Index     Value")
        print(f"{ni}      ------------------------")
        print(f"{ni}          Red  {hist_r_sky_index:5d} {hist_r_sky:9.2f}")
        print(f"{ni}        Green  {hist_g_sky_index:5d} {hist_g_sky:9.2f}")
        print(f"{ni}         Blue  {hist_b_sky_index:5d} {hist_b_sky:9.2f}")
        print("")

        # Line 924
        # Subtract value to bring sky channels equal to reference zero level.
        hist_r_sky_sub = hist_r_sky_index - zerosky_r
        hist_g_sky_sub = hist_g_sky_index - zerosky_g
        hist_b_sky_sub = hist_b_sky_index - zerosky_b

        print(f"{ni}    - Subtracted channels to align sky reference.\n")
        print(f"{ni}        Channel   Subtract     Ref")
        print(f"{ni}        --------------------------")
        print(f"{ni}            Red     {hist_r_sky_sub:6d}  {zerosky_r:6d}")
        print(f"{ni}          Green     {hist_g_sky_sub:6d}  {zerosky_g:6d}")
        print(f"{ni}           Blue     {hist_b_sky_sub:6d}  {zerosky_b:6d}")
        print("")

        rgb_sky_sub_r = hist_r_sky_index - zerosky_r
        rgb_sky_sub_g = hist_g_sky_index - zerosky_g
        rgb_sky_sub_b = hist_b_sky_index - zerosky_b

        def sub_func(ch, value):
            return (ch - value) * (65535 / (65535 - value))

        img[:, :, 0] = sub_func(img[:, :, 0], rgb_sky_sub_r)
        img[:, :, 1] = sub_func(img[:, :, 1], rgb_sky_sub_g)
        img[:, :, 2] = sub_func(img[:, :, 2], rgb_sky_sub_b)
        img[img < 0] = 0

        print_ch_moments(img, "Subtracted image.", indent=4)
        print("")

    return img


def root_stretch(
    in_img: np.ndarray,
    rootpower: int,
    rootpower2: int,
    rootiter: int,
) -> np.ndarray:
    # Line 1088
    #
    print("- Computing root stretch.")

    # Make a copy of the input image.
    img = np.ones(in_img.shape) * in_img

    for i in range(rootiter):
        if i == 0:
            # Exponent to power stretch.
            x = 1 / rootpower

        else:
            # Exponent to power stretch for iteration 2.
            x = 1 / rootpower2

        print(f"  - Iteration {i+1} of {rootiter}.")

        b = img + 1.0
        b = b / 65536
        b = 65535 * b**x

        # We are going to make the minimum 4096 out of 65535.
        b_min = int(np.min(b))
        b_min_z = max([b_min - 4095, 0])

        # Subtract the min, b_min_z, and rescale to max.
        print(f"  - Subtracting {b_min_z:8.2f} from root stretched image.")
        b = b - b_min_z
        b = b / (65535 - b_min_z)
        img = 65535 * b

        print_ch_moments(
            img, "Image stats after root stretch and subtract.", 2
        )

        # Sky level subtraction on root stretched image.
        # Line 1206.
        img = smooth_and_subtract(
            in_img=img,
            skylevelfactor=args.skylevelfactor,
            zerosky_r=args.zerosky_red,
            zerosky_g=args.zerosky_green,
            zerosky_b=args.zerosky_blue,
            indent_level=1,
        )

    return img


def s_curve(
    in_img: np.ndarray,
    scurve: int,
    skylevelfactor: float,
    zerosky_r: int,
    zerosky_g: int,
    zerosky_b: int,
) -> np.ndarray:
    print("- Computing s-curve stretch.")
    # Make a copy of the input image.
    img = np.ones(in_img.shape) * in_img

    for i in range(scurve):
        if i + 1 == 2 or i + 1 == 4:
            xfactor = 3
            # Note: This produces a crossover point (output = input) near 0.
            #       The image is brightened overall without much
            #       effect on the low end.
            xoffset = 0.22
        else:
            xfactor = 5
            # Note: This produces a crossover point (output = input) at
            #       about 1/3 max level. Above this level, image is
            #       brighter. Below this level, image is
            #       darker with higher contrast.
            xoffset = 0.42

        scurvemin = xfactor / (
            1 + np.exp(-1 * ((0 / 65535 - xoffset) * xfactor))
        ) - (
            1 - xoffset
        )  # = -0.0345159 when i=1
        scurvemax = xfactor / (
            1 + np.exp(-1 * ((65535 / 65535 - xoffset) * xfactor))
        ) - (
            1 - xoffset
        )  # = 4.15923 when i=1
        scurveminsc = scurvemin / scurvemax  # = -0.00829863 when i==1

        print(f"  - S-curve pass {i+1}")
        print(f"    xfactor     = {xfactor:4.2f}")
        print(f"    xoffset     = {xoffset:4.2f}")
        print(f"    scurvemin   = {scurvemin:4.2f}")
        print(f"    scurvemax   = {scurvemax:4.2f}")
        print(f"    scurveminsc = {scurveminsc:4.2f}")

        xo = 1 - xoffset
        sc = img / 65535
        # Now we have (img/65535 - xoffset).
        sc = sc - xoffset
        sc = sc * xfactor
        sc = sc * -1
        # Now we have exp(-1 * ((img/65535.0 - xoffset) * xfactor)).
        sc = np.exp(sc)

        # Now we have (1 + exp(-1 * ((img/65535 - xoffset) * xfactor))).
        sc = 1.0 + sc
        sc = xfactor / sc
        sc = sc - xo
        sc = sc / scurvemax

        sc = sc - scurveminsc
        sc = 65535 * sc
        img = sc / (1 - scurveminsc)

        print_ch_moments(img, f"Image stats after s-curve, pass {i+1}.", 2)

    print("\n- Subtracting sky offset from s-curve stretched image.")
    img_subtracted = smooth_and_subtract(
        img,
        skylevelfactor,
        zerosky_r,
        zerosky_g,
        zerosky_b,
    )

    return img_subtracted


def setmin(
    in_img: np.ndarray,
    setmin_r: int,
    setmin_g: int,
    setmin_b: int,
) -> np.ndarray:
    # This makes sure there are no really dark pixels, which typically happens
    # from noise or color matrix application (in the raw converter) around
    # stars showing chromatic aberration.
    print("- Applying set minimum.")
    print("  Minimum RGB levels on output:")
    print(f"    Red:   {setmin_r}")
    print(f"    Green: {setmin_g}")
    print(f"    Blue:  {setmin_b}")

    # Make a copy of the input image.
    img = np.ones(in_img.shape) * in_img

    # Keep some of the low level, noise, to keep a more natural look.
    zx = 0.2

    mask_r = img[:, :, 0] < setmin_r
    mask_g = img[:, :, 1] < setmin_g
    mask_b = img[:, :, 2] < setmin_b

    img[mask_r, 0] = setmin_r + zx * img[mask_r, 0]
    img[mask_g, 1] = setmin_g + zx * img[mask_g, 1]
    img[mask_b, 2] = setmin_b + zx * img[mask_b, 2]

    return img


def color_correct(
    in_img: np.ndarray,
    in_img_original: np.ndarray,
    colorenhance: float,
    zerosky_r: int,
    zerosky_g: int,
    zerosky_b: int,
    shape_x: int,
    shape_y: int,
):
    # Make a copy of input image.
    img = np.ones(in_img.shape) * in_img.astype(np.float64)
    img_original = np.ones(in_img_original.shape) * in_img_original.astype(
        np.float64
    )

    # Sky level subtracted to get the real zero point.
    img_original[:, :, 0] -= zerosky_r
    img_original[:, :, 1] -= zerosky_g
    img_original[:, :, 2] -= zerosky_b

    print("- Computing image ratios for color analysis.")

    # Ratios of `img_original` indicate the original color.
    # The `img` ratios are the root stretched color.
    #
    # We need to reduce the `img_original` color ratios from the
    # `img_original` ratios by the inverse of the
    # `img` color ratios.
    # If the inverse reduction is not done, then the color is overcorrected.
    #
    # In other words, the `img` color ratios show the color already
    # there in the root stretched image.
    # We reduce the `img_original` color ratios so when we apply the
    # computed color ratios (e.g. grratio) it is not overcorrected.

    # Very low number so prevents divide by 0 (out of 65535).
    img_original[img_original < 10] = 10
    img[img < 10] = 10

    # Green / Red ratio.
    gr = (img_original[:, :, 1] / img_original[:, :, 0]) / (
        img[:, :, 1] / img[:, :, 0]
    )

    # Blue / Red ratio.
    br = (img_original[:, :, 2] / img_original[:, :, 0]) / (
        img[:, :, 2] / img[:, :, 0]
    )

    # Red / Green ratio.
    rg = (img_original[:, :, 0] / img_original[:, :, 1]) / (
        img[:, :, 0] / img[:, :, 1]
    )

    # Blue / Green ratio.
    bg = (img_original[:, :, 2] / img_original[:, :, 1]) / (
        img[:, :, 2] / img[:, :, 1]
    )

    # Green / Blue ratio.
    gb = (img_original[:, :, 1] / img_original[:, :, 2]) / (
        img[:, :, 1] / img[:, :, 2]
    )

    # Red / Blue ratio.
    rb = (img_original[:, :, 0] / img_original[:, :, 2]) / (
        img[:, :, 0] / img[:, :, 2]
    )

    print("  - Setting limits for color correction.")

    # Note: Numbers > 1 desaturate.
    zmin = 0.2
    zmax = 1.0

    gr[gr < zmin] = zmin
    gr[gr > zmax] = zmax

    br[br < zmin] = zmin
    br[br > zmax] = zmax

    rg[rg < zmin] = zmin
    rg[rg > zmax] = zmax

    bg[bg < zmin] = zmin
    bg[bg > zmax] = zmax

    gb[gb < zmin] = zmin
    gb[gb > zmax] = zmax

    rb[rb < zmin] = zmin
    rb[rb > zmax] = zmax

    # Line 2034
    #
    # Only make color adjustment at the upper end and
    # proportionally less correction at lower intensities.
    print(
        "  - Color ratio images after limit set {:.2f} to {:.2f}".format(
            zmin,
            zmax,
        )
    )
    print(
        "    - Green / Red:   min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(gr),
            np.max(gr),
            np.mean(gr),
        )
    )
    print(
        "    - Blue  / Red:   min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(br),
            np.max(br),
            np.mean(br),
        )
    )
    print(
        "    - Red   / Green: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(rg),
            np.max(rg),
            np.mean(rg),
        )
    )
    print(
        "    - Blue  / Green: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(bg),
            np.max(bg),
            np.mean(bg),
        )
    )
    print(
        "    - Green / Blue:  min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(gb),
            np.max(gb),
            np.mean(gb),
        )
    )
    print(
        "    - Red   / Blue:  min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(rb),
            np.max(rb),
            np.mean(rb),
        )
    )

    print("  - Computing intensity independent color correction.")
    cavgn = ((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3) / 65535
    cavgn[cavgn < 0] = 0

    # Normalize to the maximum.
    if np.max(cavgn) < 1:
        cavgn = cavgn / np.max(cavgn)

    # `cavgn` reduces color correction as the scene darkens, so chroma noise
    # does not get enchanced. This parameter will mean undercorrection of
    # color rather than over correct, and as scene intensity is more
    # undercorrected.
    #
    # For no undercorrection, set: cavgn = 1.0.
    # For less undercorrection, decrease the exponent, like 0.1.
    # See also the cfactor value below, which does a first order
    # correction to cavgn < 1. Note, this is subjective to
    # produce pleasing colors. You can effectively change all
    # this on the command line with the -colorenhance flag.
    cavgn = cavgn**0.2

    # Prevent low level from completely being lost.
    cavgn = (cavgn + 0.3) / (1.0 + 0.3)

    # Line 2080
    print(
        "  - Color correction intensity range factor.\n"
        "    Image stats: min={:.2f}, max={:.2f}, mean={:.2f}.".format(
            np.min(cavgn),
            np.max(cavgn),
            np.mean(cavgn),
        )
    )

    # Single factor to modify color enhancement default.
    # Set a little above 1 because the average cavgn is less than 1.
    cfactor = 1.2

    # For strict colors, set cfactor = colorenhance = cavgn = 1.0.
    # That can result in enhanced color noise in the darker parts
    # of the image, depending on the S/N of the image.
    cfe = cfactor * colorenhance * cavgn

    print(
        "  - Color correction cfe range factor.\n"
        "    Image stats: min={:.2f}, max={:.2f}, mean={:.2f}.".format(
            np.min(cfe),
            np.max(cfe),
            np.mean(cfe),
        )
    )

    gr = 1 + (cfe * (gr - 1))
    br = 1 + (cfe * (br - 1))
    rg = 1 + (cfe * (rg - 1))
    bg = 1 + (cfe * (bg - 1))
    gb = 1 + (cfe * (gb - 1))
    rb = 1 + (cfe * (rb - 1))

    print("  - Color ratio images after factors applied.")
    print(
        "    - Green / Red:   min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(gr),
            np.max(gr),
            np.mean(gr),
        )
    )
    print(
        "    - Blue  / Red:   min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(br),
            np.max(br),
            np.mean(br),
        )
    )
    print(
        "    - Red   / Green: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(rg),
            np.max(rg),
            np.mean(rg),
        )
    )
    print(
        "    - Blue  / Green: min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(bg),
            np.max(bg),
            np.mean(bg),
        )
    )
    print(
        "    - Green / Blue:  min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(gb),
            np.max(gb),
            np.mean(gb),
        )
    )
    print(
        "    - Red   / Blue:  min={:.3f}, max={:.3f}, mean={:.3f}".format(
            np.min(rb),
            np.max(rb),
            np.mean(rb),
        )
    )

    print("  - Computing 6 intermediate image sets for color correction.")
    # These 6 images are computed to speed calculations below.
    c2gr = img[:, :, 1] * gr  # green adjusted
    c3br = img[:, :, 2] * br  # blue adjusted

    c1rg = img[:, :, 0] * rg  # red adjusted
    c3bg = img[:, :, 2] * bg  # blue adjusted

    c1rb = img[:, :, 0] * rb  # red adjusted
    c2gb = img[:, :, 1] * gb  # green adjusted

    # Line 2147
    print("  - Starting signal-dependent color recovery.")
    nlines = 1000
    if shape_y < 2000:
        nlines = 500
    if shape_y < 1000:
        nlines = 200
    if shape_y < 600:
        nlines = 100

    # Middle of image.
    pxmid = shape_x // 2

    for iy in range(shape_y):
        ilinex = (iy - 1) / nlines - (int((iy - 1) / int(nlines)))
        if abs(ilinex) < 0.00001:
            print(f"    Starting line {iy:4d} of {shape_y}.")

        for ix in range(shape_x):
            # Default: Red is max.
            imaxv = img[ix, iy, 0]
            imaxch = 0

            # Green is max.
            if img[ix, iy, 1] > imaxv:
                imaxv = img[ix, iy, 1]
                imaxch = 1

            # Blue is max.
            if img[ix, iy, 2] > imaxv:
                imaxv = img[ix, iy, 2]
                imaxch = 2

            # Line 2187
            #
            # Now we have which channel has the maximum value.
            if imaxch == 0:
                img[ix, iy, 1] = c2gr[ix, iy]  # green adjusted
                img[ix, iy, 2] = c3br[ix, iy]  # blue adjusted
            elif imaxch == 1:
                img[ix, iy, 0] = c1rg[ix, iy]  # red adjusted
                img[ix, iy, 2] = c3bg[ix, iy]  # blue adjusted
            else:
                img[ix, iy, 0] = c1rb[ix, iy]  # red adjusted
                img[ix, iy, 1] = c2gb[ix, iy]  # green adjusted

        if abs(ilinex) < 0.00001:
            print(
                "    - line {:6d} orig-sky  RGB: {:6d} {:6d} {:6d}".format(
                    iy,
                    int(img_original[pxmid, iy, 0]),
                    int(img_original[pxmid, iy, 1]),
                    int(img_original[pxmid, iy, 2]),
                )
            )
            print(
                "    - line {:6d} corrected RGB: {:6d} {:6d} {:6d}".format(
                    iy,
                    int(img[pxmid, iy, 0]),
                    int(img[pxmid, iy, 1]),
                    int(img[pxmid, iy, 2]),
                )
            )

    print("  - Color recovery complete.")

    return img


def post_process(img: np.ndarray) -> np.ndarray:
    img[img < 0] = 0
    img[img > 65535] = 65535

    # Re-swap axes to get original shape.
    img = np.swapaxes(img, 0, 1)

    print(
        "- output image dimensions: {:d} {:d} {:d}".format(
            img.shape[0],
            img.shape[1],
            img.shape[2],
        )
    )
    print_ch_moments(img, "Output image.")

    return img


def imwrite(img: np.ndarray, path: str) -> None:
    """
    Write image to file.

    Parameters
    ----------
    img : np.ndarray
        Image data to be written.
    path : str
        Where to write the image, including extension.

    """

    # Convert image RGB -> BGR for opencv write.
    img_bgr = np.zeros(img.shape)
    img_bgr[:, :, 0] = img[:, :, 2]
    img_bgr[:, :, 1] = img[:, :, 1]
    img_bgr[:, :, 2] = img[:, :, 0]

    cv2.imwrite(path, img_bgr.astype(np.uint16))


def main(args: argparse.Namespace) -> None:
    # Environment check and setup.
    env_setup(args)

    # Read image from file.
    img = read_img(args.image)

    # Format checking and potential scaling.
    img = format_and_scale(img)

    # Print input image moments.
    print_ch_moments(img, "Input image moments:")

    # Apply a tone curve to the image.
    # Line 557.
    if args.tone_curve:
        img = tone_curve(img)
        imshow(img, "01-tone-curve", args)

    # Subtract darkest sky level from smoothed histogram.
    # Line 697.
    img = smooth_and_subtract(
        in_img=img,
        skylevelfactor=args.skylevelfactor,
        zerosky_r=args.zerosky_red,
        zerosky_g=args.zerosky_green,
        zerosky_b=args.zerosky_blue,
    )
    imshow(img, "02-sky-subtraction", args)

    # Make a copy of the subtracted image.
    # This will be needed in the later color correction step.
    img_subtracted = np.ones(img.shape) * img

    # Apply a root stretch to the dark sky subtracted image.
    # Line 1088.
    img = root_stretch(
        in_img=img,
        rootpower=args.rootpower,
        rootpower2=args.rootpower2,
        rootiter=args.rootiter,
    )
    imshow(img, "03-root-stretch", args)

    # Apply a s-curve to improve contrast.
    # Line 1529.
    if args.s_curve:
        img = s_curve(
            in_img=img,
            scurve=args.s_curve,
            skylevelfactor=args.skylevelfactor,
            zerosky_r=args.zerosky_red,
            zerosky_g=args.zerosky_green,
            zerosky_b=args.zerosky_blue,
        )
        imshow(img, "04-s-curve", args)

    # Remove really dark pixels by adjusting for minimum.
    # Line 1919.
    if args.setmin:
        img = setmin(
            img,
            args.setmin_red,
            args.setmin_green,
            args.setmin_blue,
        )
        imshow(img, "05-setmin-01", args)

    # Perform color correction on the output image.
    # Line 1951.
    if not args.no_colorcorrection:
        img = color_correct(
            img,
            img_subtracted,
            colorenhance=args.colorenhance,
            zerosky_r=args.zerosky_red,
            zerosky_g=args.zerosky_green,
            zerosky_b=args.zerosky_blue,
            shape_x=img.shape[0],
            shape_y=img.shape[1],
        )
        imshow(img, "06-color-correction", args)

    # Remove really dark pixels (again) by adjusting for minimum.
    # Line 2263.
    if args.setmin:
        img = setmin(
            img,
            args.setmin_red,
            args.setmin_green,
            args.setmin_blue,
        )
        imshow(img, "07-setmin-02", args)

    # Perform some post-processing on the final image.
    # Line 2688.
    img = post_process(img)
    imshow(img, "08-post-processing", args, flip=False)

    # Write output image to file.
    imwrite(img, os.path.join(args.output_dir, "stretched.tif"))


if __name__ == "__main__":
    args = parse_sysargs()
    main(args)
