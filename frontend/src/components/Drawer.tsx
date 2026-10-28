import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  title: ReactNode;
  label: string; // what assistive technology announces
  meta?: ReactNode; // under the title
  footer?: ReactNode;
  onClose: () => void;
  children: ReactNode;
}

/** A panel from the right, over the page: focus stays inside, and Escape, the backdrop or ✕ close it. */
export function Drawer({ title, label, meta, footer, onClose, children }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const element = dialog.current;
    if (element && !element.open) {
      if (typeof element.showModal === "function") element.showModal();
      else element.setAttribute("open", ""); // jsdom has no showModal
    }
  }, []);

  return (
    <dialog
      ref={dialog}
      className="drawer"
      aria-label={label}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => event.target === dialog.current && onClose()} // the backdrop
    >
      <div className="drawer-body">
        <header className="drawer-head">
          <div>
            <h2>{title}</h2>
            {meta}
          </div>
          <button type="button" className="secondary close" aria-label="Close" onClick={onClose}>
            ✕
          </button>
        </header>
        {children}
        {footer && <footer className="drawer-foot">{footer}</footer>}
      </div>
    </dialog>
  );
}
