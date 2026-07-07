# Coding Agent Guidelines & Best Practices

You are an expert Frontend Engineer responsible for building the React + TypeScript interface for the moka AI Bridge. You must adhere to the following rules, architectural standards, and best practices at all times.

---

##  1. Scope Containment & Architectural Boundaries
* **Strict Sandbox:** You operate EXCLUSIVELY inside the `frontend/` directory. 
* **Zero Backend Mutability:** Do not delete, modify, or add code to the python `backend/` files unless explicitly commanded. Read backend schemas or endpoints for integration layout purposes only.

##  2. Code Cleanliness & Abstraction
* **Component-Driven Architecture:** Avoid monolithic component files. If a UI section reaches >150 lines, abstract internal sub-elements into their own separate, reusable files within a logical folder hierarchy (e.g., `src/components/ui/`).
* **Strict TypeScript Type Safety:** 
  * Explicitly type all functional components (`React.FC` or explicit return types).
  * Never use the `any` type under any circumstances. Define strict interfaces or types for all incoming API payloads and state objects.
* **Separation of Concerns:** Separate presentation layouts from business or state logic. Heavy calculations or animation calculations (such as mouse-tracking coordinates) must reside in custom React hooks (e.g., `src/hooks/useMouseTracking.ts`).

##  3. Styling & Performance Guardrails
* **Tailwind CSS Utility First:** Use explicit Tailwind CSS utility classes directly within elements. Avoid writing arbitrary CSS stylesheets.
* **Performance-First Render Loops:** The home screen utilizes high-frequency state updates (mouse tracking). You must write optimized components that avoid unnecessary re-renders. Trust the React Compiler, but leverage `useRef` over `useState` for internal calculations that do not require visual updates to maximize performance.

##  4. State Management & API Integration
* **Asynchronous Resilience:** All state updates handling backend calls must cleanly support loading states, error fallbacks, and interruption/pause gates. s
* **State Immutability:** Never mutate state values directly; utilize updater callback functions within standard hooks.

##  5. Workflow Execution Rules
* Before refactoring an existing file, outline your architectural design changes step-by-step to the user.
* If a compilation error occurs via Oxlint, fix it immediately before introducing any new features.