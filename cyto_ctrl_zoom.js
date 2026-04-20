/**
 * Ctrl+Scroll zoom for Cytoscape graphs.
 *
 * All cyto.Cytoscape components with className="cyto-ctrl-zoom" have
 * userZoomingEnabled=False by default.  This script listens for wheel events
 * and only forwards zoom when Ctrl (or Cmd on Mac) is held down.
 *
 * A small overlay hint is shown briefly when the user scrolls without Ctrl.
 */
(function () {
    "use strict";

    var HINT_TIMEOUT = 1200;  // ms to show the hint overlay

    /**
     * Get the Cytoscape core instance from inside a .cyto-ctrl-zoom wrapper.
     * dash-cytoscape renders a <div> child that Cytoscape.js binds to.
     */
    function getCyInstance(container) {
        // dash-cytoscape stores the cy ref on the inner div
        var inner = container.querySelector("div[id]");
        if (!inner) return null;
        // Cytoscape.js attaches itself as ._cyreg on the container element
        if (inner._cyreg && inner._cyreg.cy) return inner._cyreg.cy;
        // Fallback: try the container itself
        if (container._cyreg && container._cyreg.cy) return container._cyreg.cy;
        return null;
    }

    function showHint(container) {
        var existing = container.querySelector(".cyto-zoom-hint");
        if (existing) {
            clearTimeout(existing._hideTimer);
            existing._hideTimer = setTimeout(function () { existing.remove(); }, HINT_TIMEOUT);
            return;
        }
        var hint = document.createElement("div");
        hint.className = "cyto-zoom-hint";
        hint.textContent = "Hold Ctrl + Scroll to zoom";
        hint.style.cssText =
            "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
            "background:rgba(15,23,42,.75);color:#e2e8f0;padding:6px 16px;" +
            "border-radius:8px;font-size:12px;pointer-events:none;z-index:10;" +
            "transition:opacity .3s;opacity:1;white-space:nowrap;";
        container.style.position = "relative";
        container.appendChild(hint);
        hint._hideTimer = setTimeout(function () {
            hint.style.opacity = "0";
            setTimeout(function () { hint.remove(); }, 350);
        }, HINT_TIMEOUT);
    }

    function attachCtrlZoom(container) {
        if (container._ctrlZoomAttached) return;
        container._ctrlZoomAttached = true;

        container.addEventListener("wheel", function (e) {
            var cy = getCyInstance(container);
            if (!cy) return;

            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                // Manually apply zoom
                var zoomFactor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
                var rect = container.getBoundingClientRect();
                var pos = {
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top
                };
                var newZoom = cy.zoom() * zoomFactor;
                newZoom = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), newZoom));
                cy.zoom({ level: newZoom, renderedPosition: pos });
            } else {
                // Show hint — do NOT prevent default so the page scrolls normally
                showHint(container);
            }
        }, { passive: false });
    }

    // Attach to existing and future .cyto-ctrl-zoom elements
    function scan() {
        document.querySelectorAll(".cyto-ctrl-zoom").forEach(attachCtrlZoom);
    }

    // Initial scan + MutationObserver for dynamically added graphs
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", scan);
    } else {
        scan();
    }
    var observer = new MutationObserver(function () { scan(); });
    observer.observe(document.body, { childList: true, subtree: true });
})();
