from dolfin import *
import numpy as np
from typing import Tuple, Optional
import cyipopt

# --- MMA Subproblem Definition for IPOPT ---
class MMASubproblem:
    """
    Optimized MMA subproblem implementation with improved scaling and numerical stability.
    """
    def __init__(self, m: int, n: int, low: np.ndarray, upp: np.ndarray,
                 alfa: np.ndarray, beta: np.ndarray, p0: np.ndarray, q0: np.ndarray,
                 P: np.ndarray, Q: np.ndarray, a0: float, a: np.ndarray,
                 b: np.ndarray, c: np.ndarray, d: np.ndarray):
        # Store problem dimensions and parameters
        self.m = m
        self.n = n
        self.low = low.flatten()
        self.upp = upp.flatten()
        self.alfa = alfa.flatten()
        self.beta = beta.flatten()
        self.p0 = p0.flatten()
        self.q0 = q0.flatten()
        self.P = P
        self.Q = Q
        self.a0 = a0
        self.a = a.flatten()
        self.b = b.flatten()
        self.c = c.flatten()
        self.d = d.flatten()

        # Total number of variables for IPOPT: n (x) + m (y) + 1 (z)
        self.nvars = n + m + 1
        
        # Scale factors for improved numerical conditioning
        self.obj_scale = 1.0 / np.maximum(np.abs(self.p0.sum() + self.q0.sum()), 1.0)
        self.con_scale = 1.0

    def objective(self, x: np.ndarray) -> float:
        """Calculates the objective function with improved numerical stability."""
        xvars = x[:self.n]
        yvars = x[self.n:self.n + self.m]
        zvar = x[self.n + self.m]

        # Add small epsilon to prevent division by zero
        eps = 1e-10
        upp_safe = self.upp - xvars + eps
        low_safe = xvars - self.low + eps
        
        # Objective calculation with safety checks
        obj_x = np.sum(self.p0 / upp_safe + self.q0 / low_safe)
        obj_y = np.sum(self.c * yvars + 0.5 * self.d * yvars**2)
        obj_z = self.a0 * zvar

        return self.obj_scale * (obj_x + obj_y + obj_z)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculates gradient with improved numerical stability."""
        xvars = x[:self.n]
        yvars = x[self.n:self.n + self.m]
        grad = np.zeros(self.nvars)

        # Add small epsilon for numerical stability
        eps = 1e-10
        upp_safe = self.upp - xvars + eps
        low_safe = xvars - self.low + eps

        # Gradient calculation
        grad[:self.n] = self.obj_scale * (self.p0 / upp_safe**2 - self.q0 / low_safe**2)
        grad[self.n:self.n + self.m] = self.obj_scale * (self.c + self.d * yvars)
        grad[self.n + self.m] = self.obj_scale * self.a0

        return grad

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Calculates constraints with improved numerical stability."""
        xvars = x[:self.n]
        yvars = x[self.n:self.n + self.m]
        zvar = x[self.n + self.m]

        # Add small epsilon for numerical stability
        eps = 1e-10
        upp_safe = self.upp - xvars + eps
        low_safe = xvars - self.low + eps

        # Constraint calculation
        term_x = self.P @ (1 / upp_safe) + self.Q @ (1 / low_safe)
        g = self.con_scale * (term_x - self.a * zvar - yvars)

        return g

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculates Jacobian with improved numerical stability."""
        xvars = x[:self.n]
        jac = np.zeros((self.m, self.nvars))

        # Add small epsilon for numerical stability
        eps = 1e-10
        upp_safe = self.upp - xvars + eps
        low_safe = xvars - self.low + eps

        # Jacobian w.r.t. x variables
        jac[:, :self.n] = self.con_scale * (self.P / upp_safe**2 - self.Q / low_safe**2)

        # Jacobian w.r.t. y variables
        jac[:, self.n:self.n + self.m] = -self.con_scale * np.eye(self.m)

        # Jacobian w.r.t. z variable
        jac[:, self.n + self.m] = -self.con_scale * self.a

        return jac

# --- Optimized MMA Subproblem Solver ---
def mmasub_ipopt_optimized(m: int, n: int, iter: int, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                          xold1: np.ndarray, xold2: np.ndarray, f0val: float, df0dx: np.ndarray,
                          fval: np.ndarray, dfdx: np.ndarray, low: np.ndarray, upp: np.ndarray,
                          a0: float, a_mma: np.ndarray, c: np.ndarray, d: np.ndarray,
                          move: float = 0.2, asyinit: float = 0.5, asydecr: float = 0.7,
                          asyincr: float = 1.2, asymin: float = 0.01, asymax: float = 10.0,
                          raa0: float = 1e-5, albefa: float = 0.1,
                          ipopt_options: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, float,
                                                                       np.ndarray, np.ndarray, dict]:
    """
    Optimized MMA subproblem solver with improved convergence properties.
    """
    # Ensure arrays are column vectors
    xval = xval.reshape(-1, 1)
    xmin = xmin.reshape(-1, 1)
    xmax = xmax.reshape(-1, 1)
    df0dx = df0dx.reshape(-1, 1)
    fval = fval.reshape(-1, 1)
    a_mma = a_mma.reshape(-1, 1)
    c = c.reshape(-1, 1)
    d = d.reshape(-1, 1)

    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))

    # --- Improved Asymptote Calculation ---
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        xold1 = xold1.reshape(-1, 1)
        xold2 = xold2.reshape(-1, 1)
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones_like(xval)
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        
        # More conservative asymptote updates for stability
        low = xval - factor * np.maximum(0.1 * (xmax - xmin), xold1 - low)
        upp = xval + factor * np.maximum(0.1 * (xmax - xmin), upp - xold1)
        
        # Tighter bounds on asymptote movement
        lowmin = xval - asymax * (xmax - xmin)
        lowmax = xval - asymin * (xmax - xmin)
        uppmin = xval + asymin * (xmax - xmin)
        uppmax = xval + asymax * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # --- Move Limit Calculation ---
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    alfa = np.maximum(np.maximum(zzz1, zzz2), xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    beta = np.minimum(np.minimum(zzz1, zzz2), xmax)

    # --- Improved Calculation of p, q, P, Q, and b ---
    xmami = xmax - xmin
    xmami = np.maximum(xmami, 1e-5 * eeen)
    
    ux1 = upp - xval
    xl1 = xval - low
    
    # Ensure minimum distance to asymptotes
    ux1 = np.maximum(ux1, 0.01 * (xmax - xmin))
    xl1 = np.maximum(xl1, 0.01 * (xmax - xmin))

    # Improved calculation with regularization
    df0dx_plus = np.maximum(0, df0dx)
    df0dx_minus = np.maximum(0, -df0dx)
    p0 = ux1**2 * (df0dx_plus + raa0 / xmami)
    q0 = xl1**2 * (df0dx_minus + raa0 / xmami)

    dfdx_plus = np.maximum(0, dfdx)
    dfdx_minus = np.maximum(0, -dfdx)
    P = dfdx_plus * (ux1.T)
    Q = dfdx_minus * (xl1.T)
    
    # Improved constant term calculation
    b = (P @ (1/ux1) + Q @ (1/xl1)).flatten() - fval.flatten()

    # --- Create Subproblem ---
    subproblem = MMASubproblem(m, n, low, upp, alfa.flatten(), beta.flatten(), 
                              p0, q0, P, Q, a0, a_mma, b, c, d)

    # --- Improved Variable Bounds ---
    margin = 1e-6
    lb = np.concatenate([alfa.flatten() + margin, np.zeros(m), np.zeros(1)])
    ub = np.concatenate([beta.flatten() - margin, 1e20 * np.ones(m), 1e20 * np.ones(1)])
    
    # Constraint bounds
    cl = -np.inf * np.ones(m)
    cu = np.zeros(m)

    # --- Better Initial Guess ---
    x0 = np.zeros(n + m + 1)
    x0[:n] = np.clip(xval.flatten(), lb[:n] + 1e-6, ub[:n] - 1e-6)
    
    # Improved initial guess for slack variables
    g_current = subproblem.constraints(x0)
    x0[n:n+m] = np.maximum(0.0, g_current)  # Better initial slack
    x0[n+m] = 1.0

    # --- Optimized IPOPT Options ---
    if ipopt_options is None:
        ipopt_options = {
            'print_level': 0,
            'max_iter': 3000,  # Increased iteration limit
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'acceptable_iter': 15,
            'mu_strategy': 'adaptive',
            'linear_solver': 'mumps',
            'warm_start_init_point': 'yes',
            'warm_start_bound_push': 1e-8,
            'warm_start_mult_bound_push': 1e-8,
            'nlp_scaling_method': 'gradient-based',
            'obj_scaling_factor': 1.0,
            'bound_relax_factor': 1e-8,
            'honor_original_bounds': 'yes',
            'check_derivatives_for_naninf': 'yes'
        }

    # Create and solve IPOPT problem
    try:
        nlp = cyipopt.Problem(
            n=len(x0), m=len(cl),
            problem_obj=subproblem,
            lb=lb, ub=ub,
            cl=cl, cu=cu
        )

        for key, value in ipopt_options.items():
            nlp.add_option(key, value)

        x_opt, info = nlp.solve(x0)
        nlp.close()
        
    except Exception as e:
        print(f"IPOPT solver failed: {e}")
        # Fallback: return current values with minimal change
        x_opt = x0.copy()
        x_opt[:n] = xval.flatten()
        info = {'status': -1, 'status_msg': f'Solver failed: {e}'}

    # Extract results
    xmma = x_opt[:n].reshape(-1, 1)
    ymma = x_opt[n:n+m].reshape(-1, 1)
    zmma = x_opt[n+m]

    return xmma, ymma, zmma, low, upp, info


def kktcheck_improved(fval: np.ndarray, xmma: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> Tuple[float, float]:
    """Improved KKT check with better scaling."""
    # Normalize constraint violations
    constraint_violation = np.maximum(0, fval)
    max_constraint_violation = np.max(constraint_violation) if constraint_violation.size > 0 else 0.0

    # Normalize bound violations
    bound_violation_low = np.maximum(0, xmin - xmma) / (xmax - xmin + 1e-10)
    bound_violation_upp = np.maximum(0, xmma - xmax) / (xmax - xmin + 1e-10)
    max_bound_violation = np.max(np.concatenate([bound_violation_low, bound_violation_upp]))

    kkt_residual = max(max_constraint_violation, max_bound_violation)
    return kkt_residual, max_constraint_violation


# -------------------------------
# Optimized Problem Parameters
# -------------------------------
volfrac = 0.5
penal = 3.0
max_iter = 200
rmin = 0.04  # Slightly reduced for better convergence

# Material properties
E0 = 1.0
Emin = 1e-9
nu = 0.3

# -------------------------------
# Create Mesh and Function Spaces
# -------------------------------
mesh = RectangleMesh(Point(0, 0), Point(2.0, 1.0), 100, 50)  # Slightly coarser mesh for testing
V_d = FunctionSpace(mesh, "P", 1)
V_u = VectorFunctionSpace(mesh, "P", 1)

# -------------------------------
# Define design variable and filter
# -------------------------------
rho_unfiltered = Function(V_d, name="Unfiltered Density")
rho_filtered = Function(V_d, name="Filtered Density")

# Better initialization - start closer to target
rho_unfiltered.vector()[:] = volfrac
rho_filtered.vector()[:] = volfrac

n_dofs = V_d.dim()

# Improved filter setup
v_filter = TestFunction(V_d)
u_filter = TrialFunction(V_d)
a_filter = rmin**2 * inner(grad(u_filter), grad(v_filter)) * dx + u_filter * v_filter * dx
L_filter = rho_unfiltered * v_filter * dx
A_filter = assemble(a_filter)

def apply_density_filter(rho_in, rho_out):
    """Applies the Helmholtz filter."""
    b_filter = assemble(L_filter)
    solve(A_filter, rho_out.vector(), b_filter)

# -------------------------------
# Define elasticity problem
# -------------------------------
def eps(u):
    return sym(grad(u))

def sigma(u, rho):
    # Improved SIMP interpolation with better conditioning
    E_val = Emin + rho**penal * (E0 - Emin)
    mu = E_val / (2 * (1 + nu))
    lmbda = E_val * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * eps(u) + lmbda * tr(eps(u)) * Identity(2)

u = TrialFunction(V_u)
v = TestFunction(V_u)
a = inner(sigma(u, rho_filtered), eps(v)) * dx

# Define boundary conditions and load
class LoadBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.0) and near(x[1], 0.5, 0.05)

class FixedBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
LoadBoundary().mark(facets, 1)
FixedBoundary().mark(facets, 2)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

load = Constant((0.0, -1.0))
L = inner(load, v) * ds(1)

bc_fixed = DirichletBC(V_u, Constant((0.0, 0.0)), facets, 2)
bcs = [bc_fixed]

u_sol = Function(V_u, name="Displacement")

# --- Improved Sensitivity Analysis ---
def compute_compliance_and_sensitivity():
    """Optimized compliance and sensitivity computation."""
    # Solve state equation
    A, b_vec = assemble_system(a, L, bcs)
    solve(A, u_sol.vector(), b_vec)

    # Compliance - ensure we get a scalar value
    compliance_val = assemble(action(a, u_sol))
    # Convert to scalar if it's a Vector object
    if hasattr(compliance_val, 'get_local'):
        compliance_val = float(compliance_val.get_local()[0])
    elif hasattr(compliance_val, 'array'):
        compliance_val = float(compliance_val.array()[0])
    else:
        compliance_val = float(compliance_val)

    # Sensitivity computation with better numerical properties
    sens_expr = -penal * (E0 - Emin) * rho_filtered**(penal - 1) * inner(sigma(u_sol, Constant(1.0)), eps(u_sol))
    
    # Project and filter sensitivity
    sens_func = project(sens_expr, V_d)
    adjoint_rhs = assemble(sens_func * v_filter * dx)
    sens_vec = Function(V_d).vector()
    solve(A_filter, sens_vec, adjoint_rhs)
    
    return compliance_val, sens_vec.get_local()

# --- Constraint Functions ---
def volume_constraint(rho_vals):
    return np.mean(rho_vals) - volfrac

def volume_sensitivity(rho_vals):
    return np.ones_like(rho_vals) / len(rho_vals)

# -------------------------------
# Optimized Topology Optimization Main Loop
# -------------------------------
vtkfile = File("cantilever_mma_ipopt_optimized.pvd")

# Initialize MMA parameters
m = 1
n = n_dofs
xmin = np.zeros(n)
xmax = np.ones(n)
move = 0.15  # Reduced move limit for stability

# MMA history variables
xold1 = np.zeros(n)
xold2 = np.zeros(n)
low = np.zeros(n)
upp = np.zeros(n)

# Optimized MMA parameters
a0 = 1.0
a_mma = np.array([0.0])
c = np.array([1000.0])  # Reduced penalty for better conditioning
d = np.array([1.0])     # Added quadratic penalty for regularization

print("--- Starting Optimized Topology Optimization with MMA-IPOPT ---")
print(f"Number of design variables: {n}")
print(f"Target volume fraction: {volfrac}")
print("-----------------------------------------------------")

# Store history for convergence monitoring
obj_history = []
vol_history = []
change_history = []

for itr in range(max_iter):
    # Get current design variables
    xval = rho_unfiltered.vector().get_local()
    
    # Apply filter
    apply_density_filter(rho_unfiltered, rho_filtered)

    # Compute objective and sensitivity
    f0val, df0dx = compute_compliance_and_sensitivity()

    # Compute volume constraint and sensitivity
    fval = np.array([volume_constraint(xval)])
    dfdx = volume_sensitivity(xval).reshape(1, -1)

    # Store history - ensure scalar values
    vol_current = np.mean(xval)
    # Ensure f0val is a scalar
    if hasattr(f0val, 'get_local'):
        f0val_scalar = float(f0val.get_local()[0])
    elif hasattr(f0val, 'array'):
        f0val_scalar = float(f0val.array()[0])
    else:
        f0val_scalar = float(f0val)
    
    obj_history.append(f0val_scalar)
    vol_history.append(vol_current)

    # Print iteration info
    print(f"Iter {itr+1:3d}: Obj = {f0val_scalar:.4e} | Vol = {vol_current:.3f} | Constr = {fval[0]:.3e}")

    # Solve MMA subproblem with improved solver
    xmma, ymma, zmma, low_new, upp_new, info = mmasub_ipopt_optimized(
        m, n, itr + 1, xval, xmin, xmax,
        xold1, xold2, f0val_scalar, df0dx, fval, dfdx,
        low, upp, a0, a_mma, c, d, move=move
    )
    
    # Update asymptotes
    low, upp = low_new, upp_new

    # Check solver status
    if info['status'] not in [0, 1]:  # 0 = optimal, 1 = acceptable
        print(f"   IPOPT warning: {info.get('status_msg', 'Unknown status')}")

    # Update design variables
    xold2 = xold1.copy()
    xold1 = xval.copy()
    rho_unfiltered.vector().set_local(xmma.flatten())
    
    # Check convergence
    change = np.max(np.abs(xmma.flatten() - xval))
    change_history.append(change)
    
    print(f"   Change: {change:.2e} | ymma: {ymma[0,0]:.2e}")

    # Improved convergence criteria
    if itr > 10:
        # Check multiple criteria
        recent_changes = change_history[-5:]
        avg_change = np.mean(recent_changes)
        
        # Safe objective change calculation
        try:
            if len(obj_history) >= 5:
                obj_old = float(obj_history[-5])
                obj_new = float(obj_history[-1])
                if abs(obj_old) > 1e-12:  # Avoid division by very small numbers
                    obj_change = abs(obj_new - obj_old) / abs(obj_old)
                else:
                    obj_change = 1.0
            else:
                obj_change = 1.0
        except (ValueError, TypeError, IndexError):
            obj_change = 1.0
        
        if avg_change < 5e-4 and obj_change < 1e-3:
            print(f"\nConvergence achieved at iteration {itr+1}")
            print(f"Average change: {avg_change:.2e}, Objective change: {obj_change:.2e}")
            break
    
    # Save results
    if itr % 5 == 0:  # Save every 5 iterations to reduce I/O
        rho_filtered.rename("rho", "Material Density")
        vtkfile << (rho_filtered, float(itr))

print("\n--- Optimization completed! ---")
print(f"Final objective: {f0val_scalar:.4e}")
print(f"Final volume fraction: {vol_current:.3f}")
print(f"Target volume fraction: {volfrac:.3f}")

# Final save
rho_filtered.rename("rho", "Material Density")
vtkfile << (rho_filtered, float(itr))