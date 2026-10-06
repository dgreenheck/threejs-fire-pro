/**
 * What each inspector control does, one short sentence shown as its tooltip. Settings are
 * keyed by their path in the document (`smoke.density`), force options by force type
 * (`vortex.lift`), and the rest by name.
 */
export const HINTS: Readonly<Record<string, string>> = {
  // Scene.

  // Simulation.
  voxelSize: 'Size of one cell. Smaller is more detailed and slower.',
  'grid.maxVoxels': 'Most cells the simulation may compute. Past this, new areas stay empty.',
  'grid.cutoff': 'Smoke, flame and fuel thinner than this stop keeping cells active.',
  'grid.ground': 'Makes everything below the floor (y = 0) solid.',
  smokeGrid:
    'Resolution of smoke simulation and transport. Half uses 8× fewer cells, quarter 64×, with softer detail. Restarts the simulation.',
  velocityGrid:
    'Resolution of motion and pressure. Flame and heat keep the voxel size. Coarser is faster and loses detail.',
  'motion.buoyancy': 'How strongly heat rises.',
  'motion.smokeWeight': 'How strongly smoke sinks.',
  'motion.damping': 'How quickly motion slows down.',
  'motion.vorticity': 'Adds small swirls to the flow.',

  // Flame.
  'flame.lifespan':
    'Seconds for fresh flame to expire without replenishment. Longer duration also lets it produce heat and smoke for longer; it does not set how long the emitter runs.',
  'flame.heatRate':
    'Peak heat production per second, weighted by flame age. Heat is a relative simulation field that drives buoyancy and fuel ignition, not a temperature in kelvin.',
  'flame.smokeRate':
    'Peak smoke production per second, weighted by flame age. Adds simulated smoke; smoke density controls how opaque that smoke looks.',
  'flame.expansionRate':
    'Target local gas expansion per second, strongest in young flame. Also scales expansion from fuel consumption. Zero disables both contributions. The Campfire preset starts at 0.3; this is an artistic setting, not a validated stability limit.',
  'flame.cooling':
    'Exponential heat loss per second. Zero keeps heat; higher values cool faster. The Campfire preset uses 0.62, which halves heat in about 1.12 seconds without new heating.',
  'flame.color': 'Tints the flame.',
  'flame.brightness': 'How brightly the flame glows.',
  'flame.opacity':
    'Flame absorption per meter at full glow. Higher values make flames thicker and increase their emission. Does not change smoke density or combustion.',
  'flame.temperature':
    'Maximum blackbody color temperature in kelvin. Higher looks whiter; it does not change simulated heat, buoyancy or fuel ignition.',
  'flame.sootGlow': 'How much hot smoke glows.',

  // Smoke.
  'smoke.dissipation': 'How quickly smoke fades away.',
  'smoke.color': 'Smoke color.',
  'smoke.density': 'How thick smoke looks: how much light it blocks.',
  'smoke.scattering': 'Above 0, smoke glows with the light behind it; below 0, with it behind you.',
  'smoke.shadowDensity': 'How dark the shadows smoke casts on itself are.',

  // Fuel.
  'fuel.enabled': 'Lets emitters release fuel that burns where it is hot enough.',
  'fuel.ignitionHeat': 'Heat fuel needs before it catches fire.',
  'fuel.burnRate': 'How quickly burning fuel is used up.',

  // Lighting.
  'lighting.illuminateScene': 'Lets the fire light nearby surfaces.',
  'lighting.intensity': 'How strongly the fire lights its surroundings.',

  // Rendering.
  'rendering.filter': 'How smoothly cells blend. Trilinear is fastest, cubic smoothest.',
  'rendering.raySteps': 'Most samples a ray takes. More is sharper and slower.',
  'rendering.lightingDivisor':
    'Resolution of smoke lighting and shadows. Quarter uses 64× fewer cells, with softer shadows. Applies without a restart.',
  'rendering.halfResolution': 'Draws the fire at half resolution: faster, slightly softer.',

  // Objects.
  position: 'Where it sits, in meters.',
  rotation: 'How it is turned, in degrees.',

  // Emitters.
  mode: 'Continuous emits all the time; Burst emits once each time it is triggered.',
  emitting: 'Starts or stops emission.',
  'emission.flame': 'Lights flames in its area: 1 is fresh flame, 0 none.',
  'emission.heat': 'Heat added each second.',
  'emission.smoke': 'Smoke added each second.',
  'emission.fuel': 'Fuel added each second. Turns fuel on.',
  'charge.flame': 'Flame each burst lights: 1 is fresh flame, 0 none.',
  'charge.heat': 'Heat each burst adds.',
  'charge.smoke': 'Smoke each burst adds.',
  'charge.fuel': 'Fuel each burst adds. Turns fuel on.',
  'burst.radius': 'Size of the burst, in meters.',
  'burst.outwardSpeed': 'How fast the burst pushes outward, in m/s.',
  'burst.variation.period': 'Size of the uneven patches in the burst, in meters.',
  'burst.variation.strength': 'How uneven the burst is: 0 is even.',
  emitterShape: 'The shape it emits from.',
  emitterRadius: 'Size of the emitter, in meters.',
  emitterSize: 'Size of the box along each axis, in meters.',
  launchVelocity: 'Pushes what it emits in a direction.',
  'velocity.direction': 'Direction it pushes, turned with the emitter.',
  'velocity.speed': 'Speed it pushes at, in m/s.',

  // Forces.
  forceEnabled: 'Turns the force on or off.',
  forceType: 'The kind of push: wind, turbulence, vortex or radial.',
  'wind.direction': 'Direction the wind blows.',
  'wind.strength': 'How hard the wind pushes, in m/s².',
  'turbulence.strength': 'How hard the turbulence stirs.',
  'turbulence.scale': 'Size of the eddies: higher makes smaller swirls.',
  'vortex.center': 'Center of the swirl.',
  'vortex.axis': 'Axis the swirl turns around.',
  'vortex.strength': 'How fast it swirls; negative turns the other way.',
  'vortex.radius': 'How far from its axis the swirl reaches, in meters.',
  'vortex.lift': 'Upward pull along the axis.',
  'vortex.inward': 'Pull toward the axis.',
  'radial.center': 'Center it pushes from.',
  'radial.strength': 'Pushes outward; negative pulls inward.',
  'radial.radius': 'How far it reaches, in meters.',

  // Colliders.
  colliderShape: 'The shape fire and smoke flow around.',
  colliderRadius: 'Radius of the sphere, in meters.',
  colliderSize: 'Size of the box along each axis, in meters.',
};
