import os
import numpy as np
import matplotlib.pyplot as plt
import rps.robotarium as robotarium
import cv2
from scipy.spatial import ConvexHull


N = 5  # Number of robots
NUM_OBSTACLES = 5  # Number of moving obstacles
ENV_SIZE = 3.0  #environment
ENV_BOUNDS = 1.5
MAX_ITERS = 1000
DT = 0.033  # Time step (s)
K_C = 0.3  # Coverage control gain
K_O = 0.8  # Obstacle avoidance gain
V_MAX = 0.2  # Max velocity (m/s)
OBSTACLE_RADIUS = 0.2
AVOIDANCE_THRESHOLD = 0.6  # Distance threshold for avoidance (m)
COVERAGE_THRESHOLD = 0.5  # Distance for coverage metric (m)

def compute_voronoi(positions, xvals, yvals):
    partition_pts = [[] for _ in range(N)]
    for xv in xvals:
        for yv in yvals:
            pt = np.array([xv, yv])
            distances = np.linalg.norm(positions - pt, axis=1)
            closest_robot = np.argmin(distances)
            partition_pts[closest_robot].append(pt)
    for i in range(N):
        partition_pts[i] = np.vstack(partition_pts[i]) if partition_pts[i] else np.zeros((0, 2))
    return partition_pts

def compute_centroid(points):
    if points.shape[0] == 0:
        return np.array([0.0, 0.0])
    return np.mean(points, axis=0)

def compute_coverage_percentage(partition_pts, positions):
    covered_points = 0
    total_points = 0
    for i in range(N):
        for pt in partition_pts[i]:
            total_points += 1
            if np.linalg.norm(pt - positions[i]) < COVERAGE_THRESHOLD:
                covered_points += 1
    return (covered_points / total_points) * 100 if total_points > 0 else 0

def main():

    initial_conditions = np.array([
        [1.25, 1.0, 1.0, -1.0, 0.1],
        [0.25, 0.5, -0.5, -0.75, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
    r.figure.set_size_inches(10, 6)
    r.axes.set_xlim(-1.5, 1.5)
    r.axes.set_ylim(-1.5, 1.5)
    r.axes.set_aspect('equal', adjustable='box')


    xvals = np.linspace(-1.5, 1.5, 30)
    yvals = np.linspace(-1.5, 1.5, 30)

    # Moving obstacles
    moving_obstacles = [
        np.array([0.5, 0.0]), np.array([-0.5, 0.0]), np.array([0.0, 0.5]),
        np.array([0.7, -0.7]), np.array([-0.7, 0.7])
    ]
    obstacle_trajectories = [[] for _ in range(NUM_OBSTACLES)]

    # Wall obstacles
    wall_obstacles = [
        np.array([-1.6, 0.0]),  # Left wall
        np.array([1.6, 0.0]),   # Right wall
        np.array([0.0, -1.6]),  # Bottom wall
        np.array([0.0, 1.6])    # Top wall
    ]
    all_obstacles = moving_obstacles + wall_obstacles  

    trajectories = [[] for _ in range(N)]
    coverage_history = []
    region_plots = []
    trajectory_plots = [None] * N 
    color_list = ["red", "blue", "green", "magenta", "orange"]


    os.makedirs("obstacle_avoidance_project", exist_ok=True)
    video_path = "obstacle_avoidance_project/forced_avoidance_with_walls.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dpi = 100
    width = int(r.figure.get_figwidth() * dpi)
    height = int(r.figure.get_figheight() * dpi)
    fps = 5.0
    out = None

    for t in range(MAX_ITERS):
        poses = r.get_poses()
        positions = poses[:2, :].T
        for i in range(N):
            trajectories[i].append(positions[i].copy())

        moving_obstacles[0][0] = 0.5 * np.sin(t * 0.01)
        moving_obstacles[1][1] = -0.5 * np.cos(t * 0.015)
        moving_obstacles[2][0] = 0.4 * np.sin(t * 0.02)
        moving_obstacles[3][1] = 0.6 * np.cos(t * 0.01)
        moving_obstacles[4][0] = -0.7 * np.sin(t * 0.012)
        for i in range(NUM_OBSTACLES):
            obstacle_trajectories[i].append(moving_obstacles[i].copy())
        all_obstacles = moving_obstacles + wall_obstacles  # Update combined list

        #Voronoi partitions
        partition_pts = compute_voronoi(positions, xvals, yvals)
        centroids = np.array([compute_centroid(pts) for pts in partition_pts])


        dxi = np.zeros((2, N))
        for i in range(N):
            distances = [np.linalg.norm(positions[i] - obs) - OBSTACLE_RADIUS for obs in all_obstacles]
            min_dist = min(distances)
            avoidance_vec = np.zeros(2)
            for obs in all_obstacles:
                dist = np.linalg.norm(positions[i] - obs) - OBSTACLE_RADIUS
                if dist < AVOIDANCE_THRESHOLD:
                    direction = positions[i] - obs
                    avoidance_vec += K_O * direction / max(dist, 1e-3)

            # control inputs
            if min_dist < AVOIDANCE_THRESHOLD:
                dxi[:, i] = avoidance_vec  # Force movement away from obstacles/walls
            else:
                coverage_vec = K_C * (centroids[i] - positions[i])
                dxi[:, i] = coverage_vec + avoidance_vec

            # Limit velocity
            norm = np.linalg.norm(dxi[:, i])
            if norm > V_MAX:
                dxi[:, i] *= V_MAX / norm

            
            if t % 100 == 0:
                print(f"t={t}, Robot {i}: min_dist={min_dist:.3f}, dxi={dxi[:, i]}, norm={norm:.3f}")

        
        uni_vels = np.zeros((2, N))
        for i in range(N):
            theta = poses[2, i]
            v = np.cos(theta) * dxi[0, i] + np.sin(theta) * dxi[1, i]  # Linear velocity
            w = (-np.sin(theta) * dxi[0, i] + np.cos(theta) * dxi[1, i]) / 0.2  # Angular velocity
            uni_vels[0, i] = v
            uni_vels[1, i] = w
            if t % 100 == 0:
                print(f"t={t}, Robot {i}: uni_vels=[{v:.3f}, {w:.3f}]")

        
        for elem in region_plots:
            if isinstance(elem, plt.Circle):
                elem.remove()
            else:
                try:
                    elem.remove()
                except:
                    pass
        region_plots = []

        #Voronoi regions and centroids
        for i in range(N):
            pts = partition_pts[i]
            if pts.shape[0] >= 3:
                hull = ConvexHull(pts)
                hx = pts[hull.vertices, 0]
                hy = pts[hull.vertices, 1]
                hx = np.append(hx, hx[0])
                hy = np.append(hy, hy[0])
                ln, = r.axes.plot(hx, hy, color=color_list[i], linewidth=2, label=f"Robot {i}" if t == 0 else "")
                region_plots.append(ln)
                fill = r.axes.fill(hx, hy, facecolor=color_list[i], alpha=0.2)
                for patch in fill:
                    region_plots.append(patch)
            c_plot = r.axes.scatter(centroids[i][0], centroids[i][1], marker='x', s=80, c=color_list[i])
            region_plots.append(c_plot)

        # real-time trajectories
        for i in range(N):
            traj = np.array(trajectories[i])
            if trajectory_plots[i] is not None:
                trajectory_plots[i].remove()
            trajectory_plots[i], = r.axes.plot(traj[:, 0], traj[:, 1], color=color_list[i], linewidth=1, alpha=0.7)

        # Plotting moving obstacles
        for obs in moving_obstacles:
            circle = plt.Circle(obs, OBSTACLE_RADIUS, color='gray', alpha=0.5)
            r.axes.add_patch(circle)
            region_plots.append(circle)

        # Coverage metric
        coverage = compute_coverage_percentage(partition_pts, positions)
        coverage_history.append(coverage)

        # Annotation
        ann_text = f"Iteration: {t}\nCoverage: {coverage:.1f}%"
        for txt in r.axes.texts:
            txt.remove()
        r.axes.text(-0.25, 0.95, ann_text, transform=r.axes.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.5))

        if t == 0:
            r.axes.legend(bbox_to_anchor=(1.05, 1))

        # Video frame
        r.figure.canvas.draw()
        frame = np.frombuffer(r.figure.canvas.buffer_rgba(), dtype=np.uint8)
        if t == 0:
            total_pixels = frame.size // 4
            width = total_pixels // height
            if out is not None:
                out.release()
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        frame = frame.reshape(height, width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        out.write(frame)

        # Apply velocities
        r.set_velocities(np.arange(N), uni_vels)
        r.step()

    out.release()
    r.call_at_scripts_end()

    # Plot coverage history
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(coverage_history)), coverage_history, label="Forced Avoidance Coverage")
    plt.axhline(y=np.mean(coverage_history[-100:]), color='r', linestyle='--', label="Final Mean")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage Percentage (%)")
    plt.title("Multi-Robot Coverage with Forced Obstacle and Wall Avoidance")
    plt.legend()
    plt.grid(True)
    plt.savefig("obstacle_avoidance_project/coverage_plot.png", dpi=300)
    plt.close()

    # Plot trajectories
    plt.figure(figsize=(10, 6))
    for i in range(N):
        traj = np.array(trajectories[i])
        plt.plot(traj[:, 0], traj[:, 1], color=color_list[i], label=f"Robot {i}", linewidth=2)
        plt.scatter(traj[0, 0], traj[0, 1], c=color_list[i], marker='o', s=100, label=f"R{i} Start" if i == 0 else None)
        plt.scatter(traj[-1, 0], traj[-1, 1], c=color_list[i], marker='*', s=150, label=f"R{i} End" if i == 0 else None)
    for i in range(NUM_OBSTACLES):
        obs_traj = np.array(obstacle_trajectories[i])
        plt.plot(obs_traj[:, 0], obs_traj[:, 1], color='gray', linestyle='--', alpha=0.5)
        plt.scatter(obs_traj[-1, 0], obs_traj[-1, 1], c='gray', marker='o', s=50)
        circle = plt.Circle(obs_traj[-1], OBSTACLE_RADIUS, color='gray', alpha=0.3)
        plt.gca().add_patch(circle)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Robot Trajectories with Forced Obstacle and Wall Avoidance")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("obstacle_avoidance_project/trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Final Coverage: {coverage_history[-1]:.1f}%")

if __name__ == "__main__":
    main()