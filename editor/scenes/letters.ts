import { FontLoader, type FontData } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

// Space Grotesk Bold subset. These are the actual font outlines for the four
// characters used by the demo, without a runtime font request.
const typeface: FontData = {
  familyName: 'Space Grotesk',
  ascender: 984,
  descender: -292,
  underlinePosition: -100,
  underlineThickness: 50,
  boundingBox: { yMin: -274, xMin: -49, yMax: 1081, xMax: 1203 },
  resolution: 1000,
  original_font_information: {
    copyright: 'Copyright 2020 The Space Grotesk Project Authors',
    license: 'SIL Open Font License 1.1',
    weight: '700',
  },
  glyphs: {
    F: {
      x_min: 66,
      x_max: 506,
      ha: 534,
      o: 'm 66 0 l 198 0 l 198 291 l 482 291 l 482 411 l 198 411 l 198 580 l 506 580 l 506 700 l 66 700',
    },
    I: {
      x_min: 66,
      x_max: 198,
      ha: 264,
      o: 'm 66 0 l 198 0 l 198 700 l 66 700',
    },
    R: {
      x_min: 66,
      x_max: 588,
      ha: 632,
      o: 'm 66 0 l 198 0 l 198 264 l 382 264 q 423.5 249 411 264 q 436 210 436 234 l 436 0 l 568 0 l 568 229 q 546 296.5 568 271 q 484 324 524 322 l 484 342 q 558 399 528 360 q 588 501 588 438 l 588 513 q 561 612 588 570 q 485 677 534 654 q 370 700 436 700 l 66 700 m 198 384 l 198 580 l 356 580 q 430 554.5 404 580 q 456 487 456 529 l 456 477 q 429.5 409.5 456 435 q 356 384 403 384',
    },
    E: {
      x_min: 66,
      x_max: 522,
      ha: 554,
      o: 'm 66 0 l 522 0 l 522 120 l 198 120 l 198 293 l 488 293 l 488 413 l 198 413 l 198 580 l 516 580 l 516 700 l 66 700',
    },
    ' ': { x_min: 0, x_max: 0, ha: 254 },
  },
};

export function createLetterGeometry() {
  const depth = 0.38;
  const geometry = new TextGeometry('FIRE', {
    font: new FontLoader().parse(typeface),
    size: 1.75,
    depth,
    curveSegments: 10,
    bevelEnabled: true,
    bevelThickness: 0.045,
    bevelSize: 0.035,
    bevelOffset: 0,
    bevelSegments: 3,
  });
  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox!;
  geometry.translate(-(bounds.min.x + bounds.max.x) / 2, -bounds.min.y, -depth / 2);
  geometry.computeVertexNormals();
  return geometry;
}
