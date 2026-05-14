import open3d as o3d

if __name__ == "__main__":
  
    pcd = o3d.io.read_point_cloud("fused_cleaner.ply")
    print("Displaying input pointcloud ...")
    
    # small downsample if needed
    # pcd = pcd.voxel_down_sample(voxel_size=0.02)
    
    pcd.estimate_normals()
    
    # kept for comparison
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.08
    print(f"alpha={alpha:.3f}")
    print('Running alpha shapes surface reconstruction ...')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha)
    mesh.compute_triangle_normals(normalized=True)

    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw_geometries([mesh,pcd], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("alpha_shape_mesh.ply", mesh)
    
    # Poisson surface reconstruction
    # n_threads=1 is required to fix an issue on arm64 mac
    print('Running Poisson surface reconstruction ...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, n_threads=1, depth=9, scale=1.5, linear_fit=True)
    
    # clean up
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    # this will most likely create holes in the mesh meaning .is_watertight() will return False
    mesh.remove_non_manifold_edges()
    
    bad_vertex_indices = mesh.get_non_manifold_vertices()
    mesh.remove_vertices_by_index(bad_vertex_indices)
    mesh.remove_unreferenced_vertices()
    
    print(mesh.is_watertight())
    print(mesh.is_edge_manifold())

    print('Displaying reconstructed mesh ...')
    o3d.visualization.draw_geometries([mesh,pcd], mesh_show_back_face=False)
    o3d.io.write_triangle_mesh("poisson_mesh.ply", mesh)
    
    # Taubin smoothing
    mesh_taubin = mesh.filter_smooth_taubin(
        number_of_iterations=20,
        lambda_filter=0.5,
        mu=-0.53
    )
    mesh_taubin.compute_vertex_normals()
    
    print(mesh_taubin.is_watertight())
    print(mesh_taubin.is_edge_manifold())
    print(mesh_taubin.is_vertex_manifold())
    
    o3d.visualization.draw_geometries([mesh_taubin],window_name="Taubin smoothing", mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("poisson_taubin_mesh.ply", mesh_taubin)
