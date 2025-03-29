import os
import numpy as np
import matplotlib.pyplot as plt
import rps.robotarium as robotarium
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull


N = 5  # Number of robots
ENV_SIZE = 3.0  #environment
MAX_ITERS = 1000
DT = 0.033  # Time step (s)
K_C = 0.5  # Coverage control gain
K_O = 0.1  # Obstacle avoidance gain
V_MAX = 0.2  # Max velocity (m/s)
OBSTACLE_RADIUS = 0.2
COVERAGE_THRESHOLD = 0.5  # Distance for coverage metric (m)

def compute_voronoi(positions, xvals, yvals):
    vor = Voronoi(positions)
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
    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]

    # Obstacles
    obstacles = [np.array([0.5, 0.0]), np.array([-0.5, 0.0])]


    trajectories = [[] for _ in range(N)]
    coverage_history = []
    region_plots = []
    color_list = ["red", "blue", "green", "magenta", "orange"]

    os.makedirs("coverage_project", exist_ok=True)
    video_path = "coverage_project/adaptive_coverage.mp4"
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

        # Updating obstacle positions (sinusoidal motion)
        obstacles[0][1] = 0.5 * np.sin(t * 0.01)
        obstacles[1][1] = -0.5 * np.cos(t * 0.01)

        #Voronoi partitions
        partition_pts = compute_voronoi(positions, xvals, yvals)
        centroids = np.array([compute_centroid(pts) for pts in partition_pts])

        #control inputs
        dxi = np.zeros((2, N))
        for i in range(N):
            coverage_vec = K_C * (centroids[i] - positions[i])
            avoidance_vec = np.zeros(2)
            for obs in obstacles:
                dist = np.linalg.norm(positions[i] - obs) - OBSTACLE_RADIUS
                if dist < 0.5:  # Proximity threshold
                    direction = positions[i] - obs
                    avoidance_vec -= K_O * direction / max(dist, 0.01)
            dxi[:, i] = coverage_vec + avoidance_vec
            # Limit velocity
            norm = np.linalg.norm(dxi[:, i])
            if norm > V_MAX * DT:
                dxi[:, i] *= (V_MAX * DT / norm)

        uni_vels = np.zeros((2, N))
        for i in range(N):
            theta = poses[2, i]
            uni_vels[0, i] = dxi[0, i] * np.cos(theta) + dxi[1, i] * np.sin(theta)
            uni_vels[1, i] = (-dxi[0, i] * np.sin(theta) + dxi[1, i] * np.cos(theta)) / 0.2

        for elem in region_plots:
            elem.remove()
        region_plots = []
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
        for obs in obstacles:
            circle = plt.Circle(obs, OBSTACLE_RADIUS, color='gray', alpha=0.5)
            r.axes.add_patch(circle)
            region_plots.append(circle)

        coverage = compute_coverage_percentage(partition_pts, positions)
        coverage_history.append(coverage)

        ann_text = f"Iteration: {t}\nCoverage: {coverage:.1f}%"
        for txt in r.axes.texts:
            txt.remove()
        r.axes.text(-0.25, 0.95, ann_text, transform=r.axes.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.5))

        if t == 0:
            r.axes.legend(bbox_to_anchor=(1.05, 1))

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

        r.set_velocities(range(N), uni_vels)
        r.step()

    out.release()
    r.call_at_scripts_end()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(coverage_history)), coverage_history, label="Adaptive Coverage")
    plt.axhline(y=np.mean(coverage_history[-100:]), color='r', linestyle='--', label="Final Mean")
    plt.xlabel("Iteration")
    plt.ylabel("Coverage Percentage (%)")
    plt.title("Multi-Robot Adaptive Coverage Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("coverage_project/coverage_plot.png", dpi=300)
    plt.close()

    print(f"Final Coverage: {coverage_history[-1]:.1f}%")

if __name__ == "__main__":
    main()