meta = site_meta(DATA_DIR).reindex(site_codes)

# Build graph on sites with known coordinates; keep identity for missing-metadata sites.
known_mask = meta["lat"].notna() & meta["lon"].notna()
known_idx = np.where(known_mask.values)[0]
missing_idx = np.where(~known_mask.values)[0]

adj = torch.eye(N_SITES, dtype=torch.float32)
if len(known_idx) >= 2:
    adj_known = build_knn_adj(
        meta.loc[known_mask, "lat"].to_numpy(),
        meta.loc[known_mask, "lon"].to_numpy(),
        k=K_NN,
    )
    adj[np.ix_(known_idx, known_idx)] = adj_known

assert adj.shape == (N_SITES, N_SITES), (
    f"Adjacency shape {tuple(adj.shape)} does not match expected "
    f"({N_SITES}, {N_SITES})"
)

# Visualise sparsity
edges_per_site = (adj > 0).sum(dim=1).float()
print("Graph summary:")
print(f"  Nodes              : {adj.shape[0]}")
print(f"  Sites w/ coords    : {len(known_idx)}")
print(f"  Sites missing meta : {len(missing_idx)}")
print(f"  Non-zero entries   : {(adj > 0).sum().item():,}")
print(
    f"  Edges per site     : min={int(edges_per_site.min())}  "
    f"mean={edges_per_site.mean():.1f}  max={int(edges_per_site.max())}"
)

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(edges_per_site.numpy(), bins=15, color="seagreen", edgecolor="white")
ax.set_xlabel("Edges per site (including self-loop)")
ax.set_ylabel("Count")
ax.set_title(f"k-NN graph connectivity  (k={K_NN})")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()