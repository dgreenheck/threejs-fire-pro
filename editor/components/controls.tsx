import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Slider, Switch, Dialog, DropdownMenu, Tooltip } from 'radix-ui';
import { motion, useReducedMotion } from 'motion/react';
import { Check, ChevronDown, ChevronUp, X } from 'lucide-react';
export const spring = { type: 'spring' as const, stiffness: 450, damping: 28 };
/** A field's name, and what it does on hover. */
function FieldName({ label, unit = '', hint }: { label: string; unit?: string; hint?: string }) {
  const name = (
    <span className="field-name">
      {label}
      {unit && <span className="unit"> {unit}</span>}
    </span>
  );
  if (!hint) return name;
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{name}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip" side="left" sideOffset={10} collisionPadding={8}>
          {hint}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
export function Toggle({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  /** What the setting does, shown on hover. */
  hint?: string;
}) {
  const reduced = useReducedMotion();
  return (
    <label className="field-row toggle-label">
      <FieldName label={label} hint={hint} />
      <Switch.Root aria-label={label} checked={value} onCheckedChange={onChange} className="switch">
        <Switch.Thumb asChild>
          <motion.span
            className="switch-thumb"
            animate={{ x: value ? 16 : 0, scale: reduced ? 1 : [1, 1.08, 1] }}
            transition={
              reduced
                ? { duration: 0 }
                : { x: spring, scale: { duration: 0.2, times: [0, 0.45, 1] } }
            }
          />
        </Switch.Thumb>
      </Switch.Root>
    </label>
  );
}
/** Slider step: about a hundredth of the slider's span, rounded to a power of ten. */
function sliderStep(span: number) {
  return 10 ** Math.floor(Math.log10(Math.max(span, 1e-6) / 100));
}
/** One row: label, slider and number box, both covering the valid range `[min, max]`. */
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  begin,
  end,
  unit = '',
  off = false,
  integer = false,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  /** Dim the label: the setting currently has no effect. */
  off?: boolean;
  /** What the setting does, shown on hover. */
  hint?: string;
  step?: number;
  begin?: () => void;
  end?: () => void;
  unit?: string;
  /** Only whole numbers: the slider steps by at least 1 and typed decimals are rejected. */
  integer?: boolean;
}) {
  const sliderStepSize = integer
    ? Math.max(1, Math.round(step ?? sliderStep(max - min)))
    : (step ?? sliderStep(max - min));
  const accepts = (text: string) => validNumber(text, min, max, integer);
  const [dragging, setDragging] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const reduced = useReducedMotion();
  const finish = useCallback(() => {
    setDragging(false);
    end?.();
  }, [end]);
  useEffect(() => {
    if (!dragging) return;
    // A drag can leave the control or lose focus before its local pointerup fires.
    window.addEventListener('pointerup', finish, true);
    window.addEventListener('pointercancel', finish, true);
    window.addEventListener('blur', finish);
    return () => {
      window.removeEventListener('pointerup', finish, true);
      window.removeEventListener('pointercancel', finish, true);
      window.removeEventListener('blur', finish);
      end?.();
    };
  }, [dragging, finish, end]);
  const range = `${label}: ${min} to ${max}`;
  return (
    <div className={`field-row number-field ${off ? 'field-off' : ''}`}>
      <FieldName label={label} unit={unit} hint={hint} />
      <div className="field-control">
        <Slider.Root
          aria-label={label}
          className="slider"
          min={min}
          max={max}
          step={sliderStepSize}
          value={[value]}
          onPointerDown={() => {
            setDragging(true);
            begin?.();
          }}
          onPointerUp={finish}
          onPointerCancel={finish}
          onLostPointerCapture={finish}
          onValueChange={(v) => onChange(v[0])}
          onValueCommit={finish}
        >
          <Slider.Track className="slider-track">
            <Slider.Range className="slider-fill" />
          </Slider.Track>
          <Slider.Thumb className="slider-hit" aria-label={label}>
            <motion.span
              className="slider-dot"
              animate={{ scale: dragging && !reduced ? 1.25 : 1 }}
              transition={reduced ? { duration: 0 } : spring}
            />
          </Slider.Thumb>
        </Slider.Root>
        <input
          aria-label={`${label} value`}
          title={range}
          type="number"
          min={min}
          max={max}
          step={step ?? (integer ? 1 : 'any')}
          aria-invalid={text !== null && !accepts(text)}
          value={text ?? Number(value.toFixed(6))}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => {
            if (text !== null && accepts(text)) {
              onChange(Number(text));
              setText(null);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.currentTarget.blur();
            if (e.key === 'Escape') {
              setText(null);
              e.currentTarget.blur();
            }
          }}
        />
      </div>
    </div>
  );
}
/** Typed text is a number in range, and a whole one when `integer` is set. */
function validNumber(text: string, min: number, max: number, integer: boolean) {
  const n = Number(text);
  return (
    text.trim() !== '' &&
    Number.isFinite(n) &&
    n >= min &&
    n <= max &&
    (!integer || Number.isInteger(n))
  );
}
/** Arrow and scrub increment: one tenth of the value's leading digit. */
function stepFor(value: number, floor: number) {
  const magnitude = Math.abs(value);
  return magnitude < floor * 10 ? floor : 10 ** Math.floor(Math.log10(magnitude)) / 10;
}
/**
 * Number box you drag across to change (Shift for fine steps), click to type, or step
 * with the arrows and arrow keys. Steps follow the value's size, so small and large
 * values both adjust precisely.
 */
export function ScrubInput({
  label,
  value,
  onChange,
  min,
  max,
  begin,
  end,
  precision = 0.01,
  integer = false,
  stepLabel = label,
}: {
  /** Accessible name of the number box. */
  label: string;
  /** Names the arrows "Increase <stepLabel>" and "Decrease <stepLabel>". */
  stepLabel?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  begin?: () => void;
  end?: () => void;
  /** Smallest step, used near zero. */
  precision?: number;
  /** Only whole numbers: steps are at least 1 and typed decimals are rejected. */
  integer?: boolean;
}) {
  const [text, setText] = useState<string | null>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const smallest = integer ? Math.max(1, precision) : precision;
  const clamp = (n: number) => Math.min(max, Math.max(min, n));
  const round = (n: number) =>
    integer ? Math.round(n) : Number(n.toFixed(Math.max(0, -Math.floor(Math.log10(precision)))));
  const stepAt = (current: number, fine: boolean) =>
    Math.max(integer ? 1 : 0, stepFor(current, smallest) * (fine ? 0.1 : 1));
  const nudge = (direction: number, fine = false) => {
    onChange(clamp(round(value + direction * stepAt(value, fine))));
  };
  const range = `${label}: ${min} to ${max}`;
  const invalid = text !== null && !validNumber(text, min, max, integer);
  return (
    <div className={`scrub-control ${scrubbing ? 'scrubbing' : ''}`}>
      <input
        aria-label={label}
        title={`${range}. Drag to change.`}
        type="number"
        min={min}
        max={max}
        step={integer ? 1 : 'any'}
        aria-invalid={invalid}
        value={text ?? Number(value.toFixed(6))}
        onPointerDown={(e) => {
          if (document.activeElement === e.currentTarget || e.button !== 0) return;
          // A click types; a drag scrubs. Decide once the pointer moves.
          e.preventDefault();
          const input = e.currentTarget;
          const origin = e.clientX;
          let last = e.clientX;
          let current = value;
          let moved = false;
          input.setPointerCapture(e.pointerId);
          const move = (event: PointerEvent) => {
            if (!moved && Math.abs(event.clientX - origin) < 3) return;
            if (!moved) {
              moved = true;
              last = origin;
              setScrubbing(true);
              begin?.();
            }
            // Each pixel moves one step of the current value, so a drag speeds up as
            // the value grows. Shift is ten times finer.
            current = clamp(current + (event.clientX - last) * stepAt(current, event.shiftKey));
            last = event.clientX;
            onChange(round(current));
          };
          const up = () => {
            input.removeEventListener('pointermove', move);
            input.removeEventListener('pointerup', up);
            input.removeEventListener('pointercancel', up);
            if (moved) {
              setScrubbing(false);
              end?.();
            } else {
              input.focus();
              input.select();
            }
          };
          input.addEventListener('pointermove', move);
          input.addEventListener('pointerup', up);
          input.addEventListener('pointercancel', up);
        }}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          if (text === null || invalid) return;
          onChange(Number(text));
          setText(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault();
            setText(null);
            nudge(e.key === 'ArrowUp' ? 1 : -1, e.shiftKey);
          }
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') {
            setText(null);
            e.currentTarget.blur();
          }
        }}
      />
      <div className="scrub-steps">
        <button
          type="button"
          tabIndex={-1}
          aria-label={`Increase ${stepLabel}`}
          disabled={value >= max}
          onClick={(e) => nudge(1, e.shiftKey)}
        >
          <ChevronUp size={11} />
        </button>
        <button
          type="button"
          tabIndex={-1}
          aria-label={`Decrease ${stepLabel}`}
          disabled={value <= min}
          onClick={(e) => nudge(-1, e.shiftKey)}
        >
          <ChevronDown size={11} />
        </button>
      </div>
    </div>
  );
}
/** A labeled scrub box, for values that span a wide range. */
export function ScrubField({
  label,
  unit = '',
  off = false,
  hint,
  ...input
}: Omit<Parameters<typeof ScrubInput>[0], 'label'> & {
  label: string;
  unit?: string;
  /** Dim the label: the setting currently has no effect. */
  off?: boolean;
  /** What the setting does, shown on hover. */
  hint?: string;
}) {
  return (
    <div className={`field-row scrub-field ${off ? 'field-off' : ''}`}>
      <FieldName label={label} unit={unit} hint={hint} />
      <ScrubInput label={`${label} value`} stepLabel={label} {...input} />
    </div>
  );
}
/** X, Y and Z scrub boxes. Rotation shows degrees and stores radians. */
export function VectorField({
  label,
  value,
  onChange,
  rotation = false,
  min,
  max,
  begin,
  end,
  hint,
}: {
  label: string;
  value: number[];
  onChange: (v: [number, number, number]) => void;
  rotation?: boolean;
  min?: number;
  max?: number;
  begin?: () => void;
  end?: () => void;
  /** What the setting does, shown on hover. */
  hint?: string;
}) {
  const scale = rotation ? 180 / Math.PI : 1;
  return (
    <div className="field-row vector-field">
      <FieldName label={label} hint={hint} />
      <div className="vector-inputs">
        {['X', 'Y', 'Z'].map((axis, i) => (
          <label key={axis}>
            <span>{axis}</span>
            <ScrubInput
              label={`${label} ${axis}`}
              value={value[i] * scale}
              min={min ?? (rotation ? -720 : -1e5)}
              max={max ?? (rotation ? 720 : 1e5)}
              precision={rotation ? 1 : 0.01}
              begin={begin}
              end={end}
              onChange={(n) => {
                const next = [...value] as [number, number, number];
                next[i] = n / scale;
                onChange(next);
              }}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="property-section" aria-label={title}>
      <h3 className="section-title">{title}</h3>
      <div className="section-body">{children}</div>
    </section>
  );
}
export function Modal({
  title,
  description,
  open,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  description: string;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(v) => {
        if (!v) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className={`dialog ${wide ? 'dialog-wide' : ''}`}>
          <header>
            <div>
              <Dialog.Title>{title}</Dialog.Title>
              <Dialog.Description>{description}</Dialog.Description>
            </div>
            <Dialog.Close className="icon-button" aria-label="Close dialog">
              <X size={18} />
            </Dialog.Close>
          </header>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function Menu({
  label,
  icon,
  items,
}: {
  label: string;
  icon?: ReactNode;
  items: { label: string; action: () => void; disabled?: boolean; danger?: boolean }[];
}) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className={icon ? 'icon-button' : 'button subtle'} aria-label={label}>
        {icon ?? (
          <>
            {label}
            <ChevronDown size={13} />
          </>
        )}
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu" sideOffset={6}>
          {items.map((item) => (
            <DropdownMenu.Item
              key={item.label}
              className={`menu-item ${item.danger ? 'danger' : ''}`}
              disabled={item.disabled}
              onSelect={item.action}
            >
              {item.label}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
/** A button showing the current choice, which opens a menu of the choices. `label` names
 * it for assistive technology. */
export function ChoiceMenu<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  const current = options.find((option) => option.value === value);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className="choice-menu" aria-label={`${label}: ${current?.label}`}>
        <span className="choice-menu-value">{current?.label}</span>
        <ChevronDown size={13} aria-hidden="true" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu" sideOffset={6} align="end" aria-label={label}>
          <DropdownMenu.RadioGroup value={value} onValueChange={(next) => onChange(next as T)}>
            {options.map((option) => (
              <DropdownMenu.RadioItem
                key={option.value}
                value={option.value}
                className="menu-item menu-choice"
              >
                <span className="menu-check">
                  <DropdownMenu.ItemIndicator>
                    <Check size={13} />
                  </DropdownMenu.ItemIndicator>
                </span>
                {option.label}
              </DropdownMenu.RadioItem>
            ))}
          </DropdownMenu.RadioGroup>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
export function Choice({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  /** What the setting does, shown on hover. */
  hint?: string;
}) {
  return (
    <label className="field-row select-field">
      <FieldName label={label} hint={hint} />
      <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option[0].toUpperCase() + option.slice(1)}
          </option>
        ))}
      </select>
    </label>
  );
}
export function TargetCheck({
  name,
  checked,
  onChange,
}: {
  name: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="target-choice">
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="target-box">
        <Check size={12} />
      </span>
      <span>{name}</span>
    </label>
  );
}
export function ColorField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  /** What the setting does, shown on hover. */
  hint?: string;
}) {
  return (
    <label className="field-row color-field">
      <FieldName label={label} hint={hint} />
      <span className="field-control">
        <input
          type="color"
          aria-label={label}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <span className="color-code">{value}</span>
      </span>
    </label>
  );
}
