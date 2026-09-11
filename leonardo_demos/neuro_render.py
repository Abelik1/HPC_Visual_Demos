"""Headless diagram of a visitor-built brain, drawn from its real weights."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .render import font

POSITIVE = (78, 226, 255)
NEGATIVE = (250, 92, 177)


def _positions(x, top, bottom, n):
    return [(x, top + (bottom - top) * (i + .5) / n) for i in range(n)]


def draw_brain(layer_weights, activations, input_names, output_names, title="YOUR BRAIN",
               subtitle="", footer="", accent=(255, 196, 92), size=(560, 520)):
    """Render every connection of the network with its actual weight.

    ``layer_weights`` is a list of (in, out) matrices for one individual and
    ``activations`` a list of per-layer activation vectors (inputs first), so a
    sensor the network ignores reads as a dark node with faint wires.
    """
    width, height = size
    image = Image.new("RGBA", size, (4, 12, 27, 238))
    d = ImageDraw.Draw(image, "RGBA")
    d.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, outline=(*accent, 190), width=2)
    d.text((22, 16), title, font=font(17, True), fill=(236, 246, 255, 255))
    if subtitle:
        d.text((22, 42), subtitle, font=font(11), fill=(150, 192, 224, 255))
    sizes = [layer_weights[0].shape[0]] + [w.shape[1] for w in layer_weights]
    columns = len(sizes)
    left, right = 118, width - 110
    xs = [left + (right - left) * i / max(1, columns - 1) for i in range(columns)]
    top, bottom = 76, height - 58
    nodes = []
    for i, n in enumerate(sizes):
        span = (bottom - top) * min(1.0, .25 + n / 16)
        mid = (top + bottom) / 2
        nodes.append(_positions(xs[i], mid - span / 2, mid + span / 2, n))
    for stage, matrix in enumerate(layer_weights):
        matrix = np.asarray(matrix, dtype=np.float32)
        scale = max(.05, float(np.percentile(np.abs(matrix), 90)))
        # Thin wires first, strong ones on top.
        pairs = sorted(((abs(float(matrix[a, b])), a, b) for a in range(matrix.shape[0]) for b in range(matrix.shape[1])))
        for magnitude, a, b in pairs:
            strength = min(1.0, magnitude / scale)
            colour = POSITIVE if matrix[a, b] >= 0 else NEGATIVE
            d.line((*nodes[stage][a], *nodes[stage + 1][b]), fill=(*colour, int(14 + 160 * strength ** 1.5)),
                   width=1 + int(2.2 * strength))
    for layer, points in enumerate(nodes):
        values = np.asarray(activations[layer] if layer < len(activations) else np.zeros(len(points)), dtype=np.float32)
        for index, (x, y) in enumerate(points):
            value = float(np.clip(values[index] if index < len(values) else 0.0, -1, 1))
            colour = POSITIVE if value >= 0 else NEGATIVE
            glow = abs(value)
            radius = 5.5 + 3.5 * glow + (2 if layer in (0, columns - 1) else 0)
            if glow > .2:
                d.ellipse((x - radius - 5, y - radius - 5, x + radius + 5, y + radius + 5), fill=(*colour, int(40 * glow)))
            d.ellipse((x - radius, y - radius, x + radius, y + radius),
                      fill=(int(20 + colour[0] * (.25 + .75 * glow)), int(30 + colour[1] * (.25 + .75 * glow)),
                            int(45 + colour[2] * (.25 + .75 * glow)), 245),
                      outline=(225, 245, 255, 230), width=1)
        if layer == 0:
            for index, (x, y) in enumerate(points):
                label = input_names[index] if index < len(input_names) else ""
                box = d.textbbox((0, 0), label, font=font(10))
                d.text((x - 14 - (box[2] - box[0]), y - 6), label, font=font(10), fill=(160, 200, 228, 245))
        if layer == columns - 1:
            for index, (x, y) in enumerate(points):
                label = output_names[index] if index < len(output_names) else ""
                d.text((x + 16, y - 7), label, font=font(12, True), fill=(*accent, 255))
    if footer:
        d.text((22, height - 34), footer, font=font(12, True), fill=(127, 239, 255, 255))
    d.text((width - 170, height - 32), "cyan + / pink − weight", font=font(10), fill=(170, 197, 224, 220))
    return image.convert("RGB")
