import { Flame, CircleStop, Rocket, RotateCcw, Zap } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { demoActions } from '../demo-actions';
import type { SimulationDocument } from '../document';
interface Props {
  document: SimulationDocument;
  disabled: boolean;
  emit(active: boolean): void;
  launch(): void;
  detonate(reset: boolean): void;
  reset(): void;
}
export function DemoActions({ document, disabled, emit, launch, detonate, reset }: Props) {
  const actions = demoActions(document);
  const reducedMotion = useReducedMotion();
  const hasActions = actions.continuous || actions.launch || actions.burst;
  return (
    <div className="demo-actions" role="group" aria-label="Demo controls">
      {hasActions && (
        <div className="demo-action-set">
          <div className="demo-action-buttons">
            {actions.launch && (
              <button className="button primary" disabled={disabled} onClick={launch}>
                <Rocket size={17} aria-hidden="true" />
                Launch
              </button>
            )}
            {actions.burst && (
              <button
                className="button primary"
                disabled={disabled}
                onClick={() => detonate(actions.resetOnBurst)}
              >
                <Zap size={17} aria-hidden="true" />
                Detonate
              </button>
            )}
            {actions.continuous && (
              <button
                className={`button ${actions.emitting ? 'demo-stop' : 'primary'}`}
                disabled={disabled}
                aria-pressed={actions.emitting}
                onClick={() => emit(!actions.emitting)}
              >
                <motion.span
                  className="action-icon"
                  key={String(actions.emitting)}
                  initial={reducedMotion ? false : { scale: 0.65, rotate: -20 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ duration: reducedMotion ? 0 : 0.25 }}
                >
                  {actions.emitting ? (
                    <CircleStop size={17} aria-hidden="true" />
                  ) : (
                    <Flame size={17} aria-hidden="true" />
                  )}
                </motion.span>
                {actions.emitting
                  ? actions.stop || 'Stop emitters'
                  : actions.start || 'Start emitters'}
              </button>
            )}
            {(actions.launch || actions.burst) && (
              <button
                className="demo-reset"
                aria-label="Reset scene"
                title="Reset scene"
                disabled={disabled}
                onClick={reset}
              >
                <RotateCcw size={17} aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
