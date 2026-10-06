// Shared by the camera raymarch, volume lighting and surface-light reduction.
// The lifetime/heat state is normalized VFX data. These coefficients are an artist
// display mapping, while Planck/CIE spectra and volume transport remain separate.
// Burning activity is a proxy for unresolved hot absorbing particles. Use the
// same coefficient in j = sigma_a * B(T) and in Beer attenuation (Kirchhoff's
// law); adding blackbody emission with zero absorption made thick fire additive.
// The fields hold smoke, heat, flame lifetime and fuel; fuel is never drawn.
// A flame's glow, from 0 to 1, at lifetime `life`: full until the last fifth of its life,
// then fading out with it.
fn fireProGlow(life: f32) -> f32 {
  return clamp(life * 5.0, 0.0, 1.0);
}

fn fireProFlameAbsorption(field: vec4f, flameOpacity: f32) -> f32 {
  return fireProGlow(field.z) * flameOpacity;
}

fn fireProExtinction(field: vec4f, optical: vec4f, flameOpacity: f32) -> f32 {
  return max(0.0, field.x) * optical.x + fireProFlameAbsorption(field, flameOpacity);
}

fn fireProEmission(
  field: vec4f,
  smoke: vec4f,
  fire: vec4f,
  optical: vec4f,
  flameOpacity: f32,
  lut: texture_2d<f32>,
  linearSampler: sampler
) -> vec3f {
  if optical.y <= 0.0 {
    return vec3f(0);
  }
  // Smoke albedo splits its extinction into scattering and absorption. Burning
  // particles add absorption/emission only; they do not scatter ambient light.
  let absorption = (1.0 - smoke.xyz) * max(0.0, field.x) * optical.x;
  let source = absorption * smoke.w + fireProFlameAbsorption(field, flameOpacity);
  if all(source == vec3f(0)) {
    return vec3f(0);
  }
  // Heat is normalized VFX state. A saturating response retains gradients in
  // concentrated sources; the temperature setting is the asymptotic ceiling.
  let heat = max(0.0, field.y);
  let kelvin = 800.0 + heat / (heat + 2.0) * (fire.w - 800.0);
  let size = f32(textureDimensions(lut).x);
  let coordinate = (clamp((kelvin - 500.0) / 9500.0, 0.0, 1.0) * (size - 1.0) + .5) / size;
  let spectrum = textureSampleLevel(lut, linearSampler, vec2f(coordinate, .5), 0);
  let thermal = spectrum.rgb * exp2(clamp(spectrum.a, -24.0, 10.0));
  // White retains the natural spectrum. Saturated artist colors progressively
  // replace its chromaticity, without claiming to model chemical line spectra.
  let tintPeak = max(fire.x, max(fire.y, fire.z));
  let saturation = 1.0 - min(fire.x, min(fire.y, fire.z)) / max(tintPeak, .000001);
  let peak = max(thermal.x, max(thermal.y, thermal.z));
  let colored = mix(thermal * tintPeak, fire.xyz * peak, saturation);
  return colored * source * optical.y;
}
