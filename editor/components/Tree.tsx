import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Tree, type TreeApi, type NodeRendererProps } from 'react-arborist';
import { Box, Flame, Wind, ChevronRight, MoreHorizontal, Folder, Plus } from 'lucide-react';
import { Menu } from './controls';
import { SIMULATION_ID, type SimulationDocument } from '../document';
interface Item {
  id: string;
  name: string;
  kind: string;
  parent?: string;
  children?: Item[];
}
export interface ObjectTreeProps {
  document: SimulationDocument;
  selected: string;
  select: (id: string) => void;
  rename: (id: string, name: string) => void;
  remove: (id: string) => void;
  duplicate: (id: string) => void;
  reorder: (ids: string[], parent: string, index: number) => void;
  add: (kind: string) => void;
}
const TreeContext = createContext<ObjectTreeProps | null>(null);
function Row({ node, style, dragHandle }: NodeRendererProps<Item>) {
  const props = useContext(TreeContext)!;
  const selectedForce = props.document.forces.find((f) => f.id === props.selected);
  const Icon =
    node.data.kind === 'simulation'
      ? Box
      : node.data.kind === 'emitters'
        ? Flame
        : node.data.kind === 'forces'
          ? Wind
          : node.data.kind === 'group'
            ? Folder
            : Box;
  const target =
    selectedForce &&
    (selectedForce.targets === null
      ? props.document.emitters.some((e) => e.id === node.id)
      : selectedForce.targets.includes(node.id));
  return (
    <div
      ref={dragHandle}
      style={style}
      className={`tree-row ${node.isSelected ? 'selected' : ''} ${target ? 'targeted' : ''}`}
      onClick={(event) => {
        event.stopPropagation();
        if (!node.isSelected) node.select();
        if (node.data.kind !== 'group') props.select(node.id);
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
        if (fixed(node.data)) node.toggle();
        else node.edit();
      }}
    >
      <button
        className="tree-disclosure"
        tabIndex={node.isLeaf ? -1 : 0}
        aria-label={`${node.isOpen ? 'Collapse' : 'Expand'} ${node.data.name}`}
        onClick={(e) => {
          e.stopPropagation();
          node.toggle();
        }}
        style={{ visibility: node.isLeaf ? 'hidden' : 'visible' }}
      >
        <ChevronRight size={12} className={node.isOpen ? 'expanded' : ''} />
      </button>
      <Icon size={15} />
      {node.isEditing ? (
        <input
          autoFocus
          aria-label="Rename object"
          defaultValue={node.data.name}
          onFocus={(e) => e.target.select()}
          onBlur={() => node.reset()}
          onKeyDown={(e) => {
            if (e.key === 'Escape') node.reset();
            if (e.key === 'Enter') {
              node.submit(e.currentTarget.value);
            }
          }}
        />
      ) : (
        <span className="tree-name" title={node.data.name}>
          {node.data.name}
        </span>
      )}
      {!fixed(node.data) && (
        <span className="row-menu" onClick={(e) => e.stopPropagation()}>
          <Menu
            label={`Actions for ${node.data.name}`}
            icon={<MoreHorizontal size={14} />}
            items={[
              { label: 'Rename', action: () => node.edit() },
              { label: 'Duplicate', action: () => props.duplicate(node.id) },
              { label: 'Delete', action: () => props.remove(node.id), danger: true },
            ]}
          />
        </span>
      )}
    </div>
  );
}
/** Groups and the simulation itself can't be renamed, moved, duplicated or deleted. */
const fixed = (item: Item) => item.kind === 'group' || item.kind === 'simulation';
export function ObjectTree(props: ObjectTreeProps) {
  const tree = useRef<TreeApi<Item>>(null);
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 230, height: 500 });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(container.current!);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (props.selected === 'scene') tree.current?.deselectAll();
  }, [props.selected]);
  const d = props.document;
  const data: Item[] = [
    {
      id: SIMULATION_ID,
      name: 'Simulation',
      kind: 'simulation',
      children: (['emitters', 'forces', 'colliders'] as const)
        .filter((key) => d[key].length)
        .map((key) => ({
          id: `${SIMULATION_ID}/${key}`,
          name: key[0].toUpperCase() + key.slice(1),
          kind: 'group',
          children: d[key].map((item) => ({
            id: item.id,
            name: item.name,
            kind: key,
            parent: SIMULATION_ID,
          })),
        })),
    },
  ];
  return (
    <TreeContext.Provider value={props}>
      <div className="hierarchy-heading">
        <span>Hierarchy</span>
        <Menu
          label="Add object"
          icon={<Plus size={16} />}
          items={['Emitter', 'Wind', 'Turbulence', 'Vortex', 'Radial', 'Collider'].map((label) => ({
            label,
            action: () => props.add(label),
          }))}
        />
      </div>
      <aside className="object-panel">
        <div ref={container} className="tree-container">
          <Tree
            ref={tree}
            data={data}
            width={size.width}
            height={size.height}
            rowHeight={matchMedia('(pointer: coarse)').matches ? 44 : 32}
            indent={14}
            padding={0}
            selection={props.selected}
            disableMultiSelection
            disableEdit={fixed}
            disableDrag={fixed}
            disableDrop={({ parentNode, dragNodes }) =>
              dragNodes.some((n) => n.parent?.id !== parentNode.id)
            }
            onSelect={(nodes) => {
              if (nodes[0] && nodes[0].data.kind !== 'group') props.select(nodes[0].id);
            }}
            onRename={({ id, name }) => props.rename(id, name)}
            onMove={({ dragIds, parentId, index }) => props.reorder(dragIds, parentId ?? '', index)}
            onDelete={({ ids }) => ids.forEach(props.remove)}
          >
            {Row}
          </Tree>
        </div>
      </aside>
    </TreeContext.Provider>
  );
}
