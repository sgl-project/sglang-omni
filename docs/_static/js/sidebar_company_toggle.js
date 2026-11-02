// Sidebar company-group toggle for the SGLang-Omni Cookbook.
//
// Goal (mirrors the SGLang docs / Mintlify model sidebar): clicking an H2
// company heading in the sidebar must ONLY expand/collapse its H3 model
// sub-items in place -- it must not navigate, and must not change the main
// content area. Sub-item links (models) keep their normal navigation.
//
// Rendered DOM (sphinx-book-theme, with collapse_navbar: False):
//   <li class="toctree-l1 has-children">          <- + "current active" on the
//     <a class="reference internal" href="...">Qwen</a>   page's own group
//     <details open="open"><summary>...</summary><ul> ...models... </ul></details>
//   </li>
// Company groups are the only nested toctree items in the sidebar, so
// `li.toctree-l1.has-children` identifies them reliably (verified: every
// page in the build has exactly 15 such items, all company groups).
(function () {
  "use strict";

  var COMPANY_GROUP_SELECTOR = "nav.bd-links li.toctree-l1.has-children > a";
  var EXPAND_MS = 250;
  var COLLAPSE_MS = 200;
  // Fallback: if a transitionend/transitioncancel is never delivered (e.g.
  // animation interrupted by another action), clean up after this extra
  // margin so no restricted max-height is ever left behind.
  var FALLBACK_PAD = 80;

  var idSeq = 0;

  // The current page's group <li> carries "current active" (verified in the
  // built HTML: <li class="toctree-l1 current active has-children">). Also
  // defensively treat a group as active when any descendant link is the
  // current one, in case a future theme marks activity only deeper down.
  function isActiveGroup(li) {
    if (li.classList.contains("active") || li.classList.contains("current")) {
      return true;
    }
    return !!li.querySelector(".current, .active");
  }

  // One controller per company group. It owns the <details> state, all
  // animation listeners and the fallback timer, so that rapid consecutive
  // clicks always converge on a consistent open/closed state.
  function GroupController(link) {
    this.link = link;
    this.li = link.parentElement;
    this.details = this.li ? this.li.querySelector(":scope > details") : null;
    this.ul = this.details ? this.details.querySelector(":scope > ul") : null;
    this.timer = null; // fallback cleanup timer (expand or collapse)
    this.expanding = false;
    this.collapsing = false;
    this.onExpandEnd = null;
    this.onCollapseEnd = null;
    if (this.details && !this.details.id) {
      this.details.id = "sidebar-company-" + idSeq++;
    }
  }

  GroupController.prototype = {
    isOpen: function () {
      return !!(this.details && this.details.hasAttribute("open"));
    },

    // Remove all animation listeners and the fallback timer, restoring the
    // sub-list to its unrestricted static style. Safe to call at any time.
    reset: function () {
      if (this.timer) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
      this.expanding = false;
      this.collapsing = false;
      this.removeTransitionListeners();
      if (this.ul) {
        this.ul.style.maxHeight = "";
        this.ul.style.transition = "";
        this.ul.style.visibility = "";
      }
    },

    removeTransitionListeners: function () {
      if (!this.ul) return;
      if (this.onExpandEnd) {
        this.ul.removeEventListener("transitionend", this.onExpandEnd);
        this.ul.removeEventListener("transitioncancel", this.onExpandEnd);
      }
      if (this.onCollapseEnd) {
        this.ul.removeEventListener("transitionend", this.onCollapseEnd);
        this.ul.removeEventListener("transitioncancel", this.onCollapseEnd);
      }
      this.onExpandEnd = null;
      this.onCollapseEnd = null;
    },

    setAria: function () {
      if (!this.link) return;
      this.link.setAttribute("aria-expanded", this.isOpen() ? "true" : "false");
      if (this.details) {
        this.link.setAttribute("aria-controls", this.details.id || "");
      }
    },

    expand: function () {
      if (!this.details) return;
      // Cancel any pending collapse: its fallback timer must not fire later
      // and remove the "open" attribute we are about to (re-)set.
      if (this.timer) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
      this.collapsing = false;
      this.removeTransitionListeners();
      this.details.setAttribute("open", "open");
      this.setAria();
      if (!this.ul) return;
      this.expanding = true;
      var ul = this.ul;
      // Restart the grow animation from zero height.
      ul.style.transition = "none";
      ul.style.maxHeight = "0px";
      ul.style.visibility = "visible";
      void ul.offsetHeight; // force reflow so the transition below applies
      ul.style.transition =
        "max-height " + EXPAND_MS + "ms ease-in-out, visibility " + EXPAND_MS + "ms";
      var self = this;
      // Only the max-height transition marks the animation as finished: the
      // same element also transitions `visibility`, whose transitionend (or
      // transitioncancel) must not run the cleanup early.
      this.onExpandEnd = function (evt) {
        if (evt.target !== ul || evt.propertyName !== "max-height") {
          return;
        }
        self.finishExpand();
      };
      ul.addEventListener("transitionend", this.onExpandEnd);
      ul.addEventListener("transitioncancel", this.onExpandEnd);
      requestAnimationFrame(function () {
        if (self.expanding && self.ul) {
          self.ul.style.maxHeight = self.ul.scrollHeight + "px";
        }
      });
      // Fallback so the sub-list always returns to an unrestricted style.
      this.timer = window.setTimeout(function () {
        self.finishExpand();
      }, EXPAND_MS + FALLBACK_PAD);
    },

    finishExpand: function () {
      if (!this.expanding) return;
      this.expanding = false;
      if (this.timer) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
      this.removeTransitionListeners();
      if (this.ul) {
        this.ul.style.maxHeight = "";
        this.ul.style.transition = "";
        this.ul.style.visibility = "";
      }
    },

    collapse: function () {
      if (!this.details) return;
      // Cancel any in-flight expand first.
      this.finishExpand();
      this.removeTransitionListeners();
      if (!this.ul) {
        this.details.removeAttribute("open");
        this.setAria();
        return;
      }
      this.collapsing = true;
      var ul = this.ul;
      ul.style.transition = "none";
      ul.style.maxHeight = ul.scrollHeight + "px";
      void ul.offsetHeight; // force reflow so the transition below applies
      ul.style.transition =
        "max-height " + COLLAPSE_MS + "ms ease-in-out, visibility " + COLLAPSE_MS + "ms";
      var self = this;
      // Only the max-height transition marks the animation as finished: the
      // same element also transitions `visibility`, whose transitionend (or
      // transitioncancel) must not run the cleanup early.
      this.onCollapseEnd = function (evt) {
        if (evt.target !== ul || evt.propertyName !== "max-height") {
          return;
        }
        self.finishCollapse();
      };
      ul.addEventListener("transitionend", this.onCollapseEnd);
      ul.addEventListener("transitioncancel", this.onCollapseEnd);
      requestAnimationFrame(function () {
        if (self.collapsing && self.ul) {
          self.ul.style.maxHeight = "0px";
        }
      });
      this.timer = window.setTimeout(function () {
        self.finishCollapse();
      }, COLLAPSE_MS + FALLBACK_PAD);
    },

    finishCollapse: function () {
      if (!this.collapsing) return;
      this.collapsing = false;
      if (this.timer) {
        window.clearTimeout(this.timer);
        this.timer = null;
      }
      this.removeTransitionListeners();
      if (this.ul) {
        this.ul.style.maxHeight = "";
        this.ul.style.transition = "";
        this.ul.style.visibility = "";
      }
      this.details.removeAttribute("open");
      this.setAria();
    },

    // Mid-animation clicks reverse direction instead of queueing; otherwise
    // the current open state decides.
    toggle: function () {
      if (this.collapsing) {
        this.expand();
        return;
      }
      if (this.expanding) {
        this.collapse();
        return;
      }
      if (this.isOpen()) {
        this.collapse();
      } else {
        this.expand();
      }
    }
  };

  function init() {
    var links;
    try {
      links = document.querySelectorAll(COMPANY_GROUP_SELECTOR);
    } catch (err) {
      if (window.console && console.warn) {
        console.warn("sidebar company toggle: selector failed", err);
      }
      return;
    }
    for (var i = 0; i < links.length; i++) {
      // One failure in a group must not prevent the remaining groups from
      // getting their click behavior, so each group is handled on its own.
      try {
        var link = links[i];
        var li = link.parentElement;
        if (!li || !li.classList.contains("has-children")) continue;
        var details = li.querySelector(":scope > details");
        if (!details) continue;
        var ctrl = new GroupController(link);
        // On first paint keep only the current page's group expanded.
        if (!isActiveGroup(li)) {
          details.removeAttribute("open");
        }
        ctrl.setAria();
        link.addEventListener(
          "click",
          (function (c) {
            return function (evt) {
              evt.preventDefault();
              evt.stopPropagation();
              try {
                c.toggle();
              } catch (err) {
                // A single interaction failure must not spread.
              }
            };
          })(ctrl),
          false
        );
      } catch (err) {
        if (window.console && console.warn) {
          console.warn("sidebar company toggle: group init skipped", err);
        }
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, false);
  } else {
    init();
  }
})();
