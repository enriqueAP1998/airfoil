"""
Interactive Panel Method Airfoil Simulator
==========================================
Educational Streamlit app for visualizing airflow around airfoils.
Developed for IES Maristas Ourense & Universidade de Vigo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from numba import njit, prange
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Airfoil Simulator",
    page_icon="✈️",
    layout="wide"
)


@njit(parallel=True, cache=True)
def compute_induced_velocities(X_flat, Y_flat, x_panels, y_panels, S, phi, sigma, gamma, n_panels):
    """
    Compute induced velocities at field points from all panels.
    JIT-compiled for speed.
    """
    n_points = len(X_flat)
    Vx_ind = np.zeros(n_points)
    Vy_ind = np.zeros(n_points)
    
    for k in prange(n_points):
        xp = X_flat[k]
        yp = Y_flat[k]
        
        for j in range(n_panels):
            dx = xp - x_panels[j]
            dy = yp - y_panels[j]
            
            cp_j = np.cos(phi[j])
            sp_j = np.sin(phi[j])
            
            dx_loc = dx * cp_j + dy * sp_j
            dy_loc = -dx * sp_j + dy * cp_j
            
            r1_sq = max(dx_loc**2 + dy_loc**2, 1e-12)
            r2_sq = max((dx_loc - S[j])**2 + dy_loc**2, 1e-12)
            
            log_term = 0.5 * np.log(r2_sq / r1_sq)
            atan_term = np.arctan2(dy_loc, dx_loc - S[j]) - np.arctan2(dy_loc, dx_loc)
            
            ul = (sigma[j] * log_term + gamma * atan_term) / (2.0 * np.pi)
            vl = (sigma[j] * atan_term - gamma * log_term) / (2.0 * np.pi)
            
            Vx_ind[k] += ul * cp_j - vl * sp_j
            Vy_ind[k] += ul * sp_j + vl * cp_j
    
    return Vx_ind, Vy_ind


class PanelMethod:
    """
    Source-Vortex Panel Method for inviscid flow around airfoils.
    """
    
    def __init__(self, naca_code, alpha_deg, n_panels=120):
        self.naca = str(naca_code)
        self.alpha_deg = alpha_deg
        self.alpha_rad = np.radians(alpha_deg)
        self.n_panels = n_panels
        
        # Generate airfoil geometry
        x_raw, y_raw = self.get_naca_coords()
        
        # Rotate geometry for angle of attack
        c_a = np.cos(self.alpha_rad)
        s_a = np.sin(self.alpha_rad)
        self.x = x_raw * c_a - y_raw * s_a
        self.y = x_raw * s_a + y_raw * c_a
        
        # Freestream is horizontal
        self.alpha_flow = 0.0
        
        # Setup panels and solve
        self.xc, self.yc, self.S, self.phi, self.beta = self.setup_panels()
        self.sigma, self.gamma = self.solve_system()
        
        # Compute aerodynamic coefficients
        self.Cp_surface, self.Vt_surface = self.compute_surface_cp()
        self.CL, self.Cd = self.compute_coefficients()
    
    def get_naca_coords(self):
        """Generate NACA 4-digit airfoil coordinates"""
        m = int(self.naca[0]) / 100.0
        p = int(self.naca[1]) / 10.0
        t = int(self.naca[2:]) / 100.0
        
        theta = np.linspace(0, np.pi, self.n_panels // 2 + 1)
        x = 0.5 * (1 - np.cos(theta))
        
        yt = 5 * t * (0.2969 * np.sqrt(x + 1e-10) - 0.1260 * x - 
                      0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        if p > 0:
            idx1 = x <= p
            yc[idx1] = (m / p**2) * (2 * p * x[idx1] - x[idx1]**2)
            dyc_dx[idx1] = (2 * m / p**2) * (p - x[idx1])
            idx2 = x > p
            yc[idx2] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[idx2] - x[idx2]**2)
            dyc_dx[idx2] = (2 * m / (1 - p)**2) * (p - x[idx2])
        
        theta_c = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta_c)
        yu = yc + yt * np.cos(theta_c)
        xl = x + yt * np.sin(theta_c)
        yl = yc - yt * np.cos(theta_c)
        
        X = np.concatenate((xu[::-1], xl[1:]))
        Y = np.concatenate((yu[::-1], yl[1:]))
        return X, Y
    
    def setup_panels(self):
        """Setup panel geometry"""
        xc = (self.x[:-1] + self.x[1:]) / 2
        yc = (self.y[:-1] + self.y[1:]) / 2
        dx = self.x[1:] - self.x[:-1]
        dy = self.y[1:] - self.y[:-1]
        S = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        beta = phi + np.pi / 2
        return xc, yc, S, phi, beta
    
    def get_influence_coeff(self, X, Y, panel_idx):
        """Get influence coefficients for a panel at point (X, Y)"""
        i = panel_idx
        dx = X - self.x[i]
        dy = Y - self.y[i]
        
        dx_local = dx * np.cos(self.phi[i]) + dy * np.sin(self.phi[i])
        dy_local = -dx * np.sin(self.phi[i]) + dy * np.cos(self.phi[i])
        
        r1_sq = np.maximum(dx_local**2 + dy_local**2, 1e-12)
        r2_sq = np.maximum((dx_local - self.S[i])**2 + dy_local**2, 1e-12)
        
        log_term = 0.5 * np.log(r2_sq / r1_sq)
        atan_term = np.arctan2(dy_local, dx_local - self.S[i]) - np.arctan2(dy_local, dx_local)
        
        u_s = (1 / (2 * np.pi)) * log_term
        v_s = (1 / (2 * np.pi)) * atan_term
        return u_s, v_s, v_s, -u_s
    
    def solve_system(self):
        """Solve for source strengths and vortex strength"""
        N = len(self.xc)
        A_source = np.zeros((N, N))
        A_vortex = np.zeros((N, N))
        
        sina = np.sin(self.alpha_flow)
        cosa = np.cos(self.alpha_flow)
        
        for i in range(N):
            nx_i = np.cos(self.beta[i])
            ny_i = np.sin(self.beta[i])
            
            for j in range(N):
                us, vs, uv, vv = self.get_influence_coeff(self.xc[i], self.yc[i], j)
                
                cp_j = np.cos(self.phi[j])
                sp_j = np.sin(self.phi[j])
                Vxs = us * cp_j - vs * sp_j
                Vys = us * sp_j + vs * cp_j
                Vxv = uv * cp_j - vv * sp_j
                Vyv = uv * sp_j + vv * cp_j
                
                A_source[i, j] = Vxs * nx_i + Vys * ny_i
                A_vortex[i, j] = Vxv * nx_i + Vyv * ny_i
        
        b = -1.0 * (cosa * np.cos(self.beta) + sina * np.sin(self.beta))
        
        # Kutta condition
        t1x, t1y = np.cos(self.phi[0]), np.sin(self.phi[0])
        tNx, tNy = np.cos(self.phi[-1]), np.sin(self.phi[-1])
        
        Vt_s0 = np.zeros(N)
        Vt_sN = np.zeros(N)
        Vt_v0 = np.zeros(N)
        Vt_vN = np.zeros(N)
        
        for j in range(N):
            us0, vs0, uv0, vv0 = self.get_influence_coeff(self.xc[0], self.yc[0], j)
            usN, vsN, uvN, vvN = self.get_influence_coeff(self.xc[-1], self.yc[-1], j)
            
            cp_j = np.cos(self.phi[j])
            sp_j = np.sin(self.phi[j])
            
            Vxs0 = us0 * cp_j - vs0 * sp_j
            Vys0 = us0 * sp_j + vs0 * cp_j
            Vxv0 = uv0 * cp_j - vv0 * sp_j
            Vyv0 = uv0 * sp_j + vv0 * cp_j
            
            VxsN = usN * cp_j - vsN * sp_j
            VysN = usN * sp_j + vsN * cp_j
            VxvN = uvN * cp_j - vvN * sp_j
            VyvN = uvN * sp_j + vvN * cp_j
            
            Vt_s0[j] = Vxs0 * t1x + Vys0 * t1y
            Vt_sN[j] = VxsN * tNx + VysN * tNy
            Vt_v0[j] = Vxv0 * t1x + Vyv0 * t1y
            Vt_vN[j] = VxvN * tNx + VyvN * tNy
        
        A_final = np.zeros((N + 1, N + 1))
        A_final[0:N, 0:N] = A_source
        A_final[0:N, N] = np.sum(A_vortex, axis=1)
        A_final[N, 0:N] = Vt_s0 + Vt_sN
        A_final[N, N] = np.sum(Vt_v0 + Vt_vN)
        
        b_final = np.zeros(N + 1)
        b_final[0:N] = b
        b_final[N] = -((np.cos(self.alpha_flow) * t1x + np.sin(self.alpha_flow) * t1y) + 
                       (np.cos(self.alpha_flow) * tNx + np.sin(self.alpha_flow) * tNy))
        
        results = np.linalg.solve(A_final, b_final)
        return results[0:N], results[N]
    
    def compute_surface_cp(self):
        """Compute Cp on surface panels"""
        Vt_surface = np.zeros(len(self.xc))
        cosa = np.cos(self.alpha_flow)
        sina = np.sin(self.alpha_flow)
        
        for i in range(len(self.xc)):
            tx = np.cos(self.phi[i])
            ty = np.sin(self.phi[i])
            
            Vt = cosa * tx + sina * ty
            
            for j in range(len(self.xc)):
                us, vs, uv, vv = self.get_influence_coeff(self.xc[i], self.yc[i], j)
                cp_j = np.cos(self.phi[j])
                sp_j = np.sin(self.phi[j])
                
                Vxs = us * cp_j - vs * sp_j
                Vys = us * sp_j + vs * cp_j
                Vxv = uv * cp_j - vv * sp_j
                Vyv = uv * sp_j + vv * cp_j
                
                Vt += self.sigma[j] * (Vxs * tx + Vys * ty)
                Vt += self.gamma * (Vxv * tx + Vyv * ty)
            
            Vt_surface[i] = Vt
        
        Cp_surface = 1 - Vt_surface**2
        return Cp_surface, Vt_surface
    
    def compute_coefficients(self):
        """Compute lift and drag coefficients"""
        # Lift coefficient from Kutta-Joukowski theorem
        # CL = 2 * gamma (for unit chord and unit freestream velocity)
        CL = 2 * self.gamma
        
        # For inviscid flow, pressure drag is zero (d'Alembert's paradox)
        # We estimate friction drag using a flat plate correlation
        # Cd_friction ≈ 1.328 / sqrt(Re) for laminar flow
        # Using Re ~ 1e6 as typical value
        Re = 1e6
        Cd_friction = 1.328 / np.sqrt(Re)
        
        # Add small induced drag component from pressure integration
        # Integrate Cp * n_x over the surface (pressure drag)
        Cd_pressure = 0.0
        for i in range(len(self.xc)):
            nx = np.cos(self.beta[i])  # Normal vector x-component
            Cd_pressure += self.Cp_surface[i] * self.S[i] * nx
        
        Cd = Cd_friction * 2 + abs(Cd_pressure)  # Factor of 2 for both surfaces
        
        return CL, Cd
    
    def compute_flow_field(self, x_bounds=(-0.5, 1.5), y_bounds=(-0.8, 0.8), res=100):
        """Compute velocity field on a grid"""
        Xg, Yg = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], res),
            np.linspace(y_bounds[0], y_bounds[1], res)
        )
        
        Vx = np.ones_like(Xg) * np.cos(self.alpha_flow)
        Vy = np.ones_like(Xg) * np.sin(self.alpha_flow)
        
        X_flat = Xg.flatten().astype(np.float64)
        Y_flat = Yg.flatten().astype(np.float64)
        
        Vx_ind, Vy_ind = compute_induced_velocities(
            X_flat, Y_flat,
            self.x[:-1].astype(np.float64),
            self.y[:-1].astype(np.float64),
            self.S.astype(np.float64),
            self.phi.astype(np.float64),
            self.sigma.astype(np.float64),
            float(self.gamma),
            len(self.xc)
        )
        
        Vx += Vx_ind.reshape(Xg.shape)
        Vy += Vy_ind.reshape(Xg.shape)
        
        path = Path(np.column_stack((self.x, self.y)))
        mask = path.contains_points(
            np.column_stack((X_flat, Y_flat))
        ).reshape(Xg.shape)
        
        Vx[mask] = 0
        Vy[mask] = 0
        
        return Xg, Yg, Vx, Vy, mask


def create_visualization(solver, show_cp=True, show_velocity=True, show_streamlines=True):
    """Create matplotlib visualization"""
    x_bounds = (-0.4, 1.4)
    y_bounds = (-0.6, 0.6)
    res = 120
    
    Xg, Yg, Vx, Vy, mask = solver.compute_flow_field(x_bounds, y_bounds, res)
    
    V_mag = np.sqrt(Vx**2 + Vy**2)
    Cp = 1 - V_mag**2
    Cp[mask] = np.nan
    V_mag[mask] = np.nan
    
    # Count active plots
    active_plots = sum([show_cp, show_velocity, show_streamlines])
    if active_plots == 0:
        return None
    
    # Create appropriate subplot layout
    if active_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    elif active_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    plot_idx = 0
    
    # Pressure Coefficient
    if show_cp:
        ax = axes[plot_idx] if active_plots > 1 else axes
        levels_cp = np.linspace(-2.5, 1.0, 50)
        cmap_cp = plt.get_cmap('RdYlBu_r').copy()
        cmap_cp.set_bad('grey')
        
        cf = ax.contourf(Xg, Yg, Cp, levels=levels_cp, cmap=cmap_cp, extend='both')
        ax.fill(solver.x, solver.y, color='lightgrey', zorder=10)
        ax.plot(solver.x, solver.y, 'k-', linewidth=1.5, zorder=11)
        plt.colorbar(cf, ax=ax, label='$C_p$', shrink=0.8)
        ax.set_title('Pressure Coefficient $C_p$', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plot_idx += 1
    
    # Velocity Magnitude
    if show_velocity:
        ax = axes[plot_idx] if active_plots > 1 else axes
        levels_v = np.linspace(0, 2.0, 40)
        cf = ax.contourf(Xg, Yg, V_mag, levels=levels_v, cmap='jet', extend='max')
        ax.fill(solver.x, solver.y, color='lightgrey', zorder=10)
        ax.plot(solver.x, solver.y, 'k-', linewidth=1.5, zorder=11)
        plt.colorbar(cf, ax=ax, label='$|V|/V_\\infty$', shrink=0.8)
        ax.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plot_idx += 1
    
    # Streamlines
    if show_streamlines:
        ax = axes[plot_idx] if active_plots > 1 else axes
        levels_cp = np.linspace(-2.5, 1.0, 50)
        cmap_cp = plt.get_cmap('RdYlBu_r').copy()
        cmap_cp.set_bad('grey')
        
        ax.contourf(Xg, Yg, Cp, levels=levels_cp, cmap=cmap_cp, extend='both', alpha=0.6)
        ax.streamplot(Xg, Yg, Vx, Vy, density=2.0, color='k', linewidth=0.5, arrowsize=0.6)
        ax.fill(solver.x, solver.y, color='lightgrey', zorder=10)
        ax.plot(solver.x, solver.y, 'k-', linewidth=1.5, zorder=11)
        ax.set_title('Streamlines', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
    
    fig.suptitle(f'NACA {solver.naca} at α = {solver.alpha_deg}°', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ==============================================================================
# STREAMLIT APP
# ==============================================================================

def main():
    # Header with logos/institutions
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✈️ Airfoil Flow Simulator")
        st.markdown("### Interactive Aerodynamics Education Tool")
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    st.sidebar.subheader("Airfoil Configuration")
    
    # NACA code input
    naca_digit1 = st.sidebar.slider("Maximum Camber (%)", 0, 9, 4, 
                                     help="First digit: Maximum camber as percentage of chord")
    naca_digit2 = st.sidebar.slider("Camber Position (×10%)", 0, 9, 4,
                                     help="Second digit: Position of maximum camber (×10% from leading edge)")
    naca_thickness = st.sidebar.slider("Thickness (%)", 6, 24, 12,
                                        help="Last two digits: Maximum thickness as percentage of chord")
    
    naca_code = f"{naca_digit1}{naca_digit2}{naca_thickness:02d}"
    st.sidebar.info(f"**NACA {naca_code}**")
    
    # Angle of attack
    alpha_deg = st.sidebar.slider("Angle of Attack (α)", -10.0, 20.0, 8.0, 0.5,
                                   help="Angle between airfoil and incoming airflow")
    
    # Panel count
    n_panels = st.sidebar.slider("Number of Panels", 60, 200, 120, 10,
                                  help="More panels = more accurate (but slower)")
    
    st.sidebar.subheader("Visualization Options")
    show_cp = st.sidebar.checkbox("Pressure Coefficient", True)
    show_velocity = st.sidebar.checkbox("Velocity Magnitude", True)
    show_streamlines = st.sidebar.checkbox("Streamlines", True)
    
    # Run simulation
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        st.session_state.run_sim = True
        st.session_state.naca = naca_code
        st.session_state.alpha = alpha_deg
        st.session_state.n_panels = n_panels
    
    # Main content area
    col_main, col_results = st.columns([3, 1])
    
    with col_main:
        # Run simulation if requested
        if 'run_sim' in st.session_state and st.session_state.run_sim:
            with st.spinner("Computing airflow... This may take a moment."):
                try:
                    solver = PanelMethod(
                        st.session_state.naca,
                        st.session_state.alpha,
                        st.session_state.n_panels
                    )
                    st.session_state.solver = solver
                    st.session_state.run_sim = False
                except Exception as e:
                    st.error(f"Error in simulation: {e}")
                    st.session_state.run_sim = False
        
        # Display results if available
        if 'solver' in st.session_state:
            solver = st.session_state.solver
            
            fig = create_visualization(solver, show_cp, show_velocity, show_streamlines)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
    
    with col_results:
        st.subheader("📊 Results")
        
        if 'solver' in st.session_state:
            solver = st.session_state.solver
            
            # Display CL and Cd
            st.metric("Lift Coefficient (CL)", f"{solver.CL:.4f}")
            st.metric("Drag Coefficient (Cd)", f"{solver.Cd:.6f}")
            
            # Lift-to-drag ratio
            if solver.Cd > 0:
                ld_ratio = solver.CL / solver.Cd
                st.metric("L/D Ratio", f"{ld_ratio:.1f}")
            
            # Additional info
            st.markdown("---")
            st.caption(f"**Circulation:** Γ = {solver.gamma:.4f}")
            st.caption(f"**Panels used:** {len(solver.xc)}")
        else:
            st.info("👆 Configure and run simulation to see results")
    
    # Educational section
    st.markdown("---")
    st.header("📚 How Does an Airplane Wing Work?")
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        st.subheader("🌬️ What is an Airfoil?")
        st.markdown("""
        An **airfoil** is the cross-sectional shape of a wing. When air flows around it, 
        something amazing happens:
        
        - **Air moves faster** over the curved top surface
        - **Air moves slower** under the flatter bottom surface
        - This speed difference creates **lower pressure on top** and **higher pressure below**
        - The pressure difference pushes the wing **UP** - this is called **LIFT**!
        
        Think of it like squeezing a slippery watermelon seed between your fingers - 
        it shoots out because of the pressure difference!
        """)
        
        st.subheader("🔢 What do the NACA Numbers Mean?")
        st.markdown("""
        The **NACA 4-digit code** describes the airfoil shape:
        
        | Digit | Meaning | Example (NACA 4412) |
        |-------|---------|---------------------|
        | 1st | Max camber (%) | 4% camber |
        | 2nd | Camber position (×10%) | 40% from leading edge |
        | 3rd-4th | Max thickness (%) | 12% thick |
        
        **Camber** is how curved the airfoil is. More camber = more lift!
        """)
    
    with col_edu2:
        st.subheader("📈 Understanding the Coefficients")
        st.markdown("""
        **Lift Coefficient (CL):**
        - Tells us how much lift the airfoil generates
        - Higher CL = more lift for the same speed
        - Changes with angle of attack (α)
        - Typical range: 0 to ~1.5 for normal flight
        
        **Drag Coefficient (Cd):**
        - Tells us how much the air resists the motion
        - Lower Cd = less fuel needed to fly
        - We want this as small as possible!
        - Typical range: 0.005 to 0.05
        
        **L/D Ratio (Lift-to-Drag):**
        - The "efficiency" of the airfoil
        - Higher is better! Gliders can have L/D > 40
        - Commercial planes: L/D ~ 15-20
        """)
        
        st.subheader("🎯 Angle of Attack (α)")
        st.markdown("""
        The angle between the wing and the incoming air:
        
        - **Small angle (0-5°):** Cruise flight, efficient
        - **Medium angle (5-12°):** Takeoff and landing
        - **Large angle (>15°):** Risk of **stall** (wing stops working!)
        
        Try changing α in the simulator and watch what happens to CL!
        """)
    
    st.markdown("---")
    st.subheader("🧪 Try These Experiments!")
    st.markdown("""
    1. **Effect of angle:** Set NACA 2412 and change α from 0° to 15°. Watch CL increase!
    2. **Effect of camber:** Compare NACA 0012 (symmetric) vs NACA 4412 (cambered) at α=5°
    3. **Effect of thickness:** Compare NACA 4408 (thin) vs NACA 4418 (thick)
    4. **Find the best L/D:** What angle gives the highest lift-to-drag ratio?
    """)
    
    # Footer with institutions
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns([1, 2, 1])
    
    with col_foot2:
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        <h4>Developed for Educational Purposes</h4>
        <p>
        <strong>🏫 IES Maristas Ourense</strong><br>
        <a href="https://www.maristasourense.com" target="_blank">www.maristasourense.com</a>
        </p>
        <p>
        <strong>🎓 Universidade de Vigo</strong><br>
        <a href="https://www.uvigo.gal" target="_blank">www.uvigo.gal</a>
        </p>
        <p style='font-size: 0.8em; margin-top: 20px;'>
        This simulator uses the Panel Method to solve potential flow equations.<br>
        The method divides the airfoil surface into small panels and calculates<br>
        how air flows around them using mathematical equations.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("© 2025 - Educational Aerodynamics Simulator | Panel Method Implementation")


if __name__ == "__main__":
    main()
