"""Tests for graphify/cache.py."""
import pytest
from pathlib import Path
from graphify.cache import (
    file_hash, cache_dir, load_cached, save_cached, cached_files, clear_cache,
    _body_content, _git_root, _git_common_dir,
)


@pytest.fixture(autouse=True)
def _clear_git_lookup_caches():
    """lru_cache on _git_root/_git_common_dir persists between tests; clear it."""
    _git_root.cache_clear()
    _git_common_dir.cache_clear()
    yield
    _git_root.cache_clear()
    _git_common_dir.cache_clear()


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    return f


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path


def test_file_hash_consistent(tmp_file):
    """Same file gives same hash on repeated calls."""
    h1 = file_hash(tmp_file)
    h2 = file_hash(tmp_file)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64  # SHA256 hex digest length


def test_file_hash_changes(tmp_path):
    """Different file contents give different hashes."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content one")
    f2.write_text("content two")
    assert file_hash(f1) != file_hash(f2)


def test_cache_roundtrip(tmp_file, cache_root):
    """Save then load returns the same result dict."""
    result = {"nodes": [{"id": "n1", "label": "Node1"}], "edges": []}
    save_cached(tmp_file, result, root=cache_root)
    loaded = load_cached(tmp_file, root=cache_root)
    assert loaded == result


def test_cache_miss_on_change(tmp_file, cache_root):
    """After file content changes, load_cached returns None."""
    result = {"nodes": [], "edges": [{"source": "a", "target": "b"}]}
    save_cached(tmp_file, result, root=cache_root)
    # Modify the file
    tmp_file.write_text("completely different content")
    assert load_cached(tmp_file, root=cache_root) is None


def test_cached_files(tmp_path, cache_root):
    """cached_files returns the set of cached hashes."""
    f1 = tmp_path / "file1.py"
    f2 = tmp_path / "file2.py"
    f1.write_text("alpha")
    f2.write_text("beta")

    save_cached(f1, {"nodes": [], "edges": []}, root=cache_root)
    save_cached(f2, {"nodes": [], "edges": []}, root=cache_root)

    hashes = cached_files(cache_root)
    assert file_hash(f1, cache_root) in hashes
    assert file_hash(f2, cache_root) in hashes


def test_clear_cache(tmp_file, cache_root):
    """clear_cache removes all .json files from the cache directory."""
    save_cached(tmp_file, {"nodes": [], "edges": []}, root=cache_root)
    cdir = cache_dir(cache_root)
    assert len(list(cdir.glob("*.json"))) > 0
    clear_cache(cache_root)
    assert len(list(cdir.glob("*.json"))) == 0


def test_cache_dir_falls_back_when_not_in_git_repo(tmp_path):
    """Outside a git repo, cache_dir returns <root>/graphify-out/cache/ (legacy)."""
    d = cache_dir(tmp_path)
    assert d == tmp_path / "graphify-out" / "cache"
    assert d.exists()


def test_cache_dir_uses_git_common_dir(tmp_path):
    """Inside a git repo (regular clone), cache_dir returns <.git>/graphify-cache/."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # simulate a regular clone
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    d = cache_dir(repo)
    assert d == repo / ".git" / "graphify-cache"
    assert d.exists()


def test_cache_dir_shared_between_main_and_worktree(tmp_path):
    """Main tree and worktree resolve to the SAME cache directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    # Create the per-worktree git dir that a `git worktree add` would create
    wt_git = git_dir / "worktrees" / "feature"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feature\n")
    # The worktree itself (separate directory, .git is a *file*)
    worktree = tmp_path / "worktrees" / "feature"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {wt_git}\n")

    main_cache = cache_dir(repo)
    wt_cache = cache_dir(worktree)

    assert main_cache == wt_cache == git_dir / "graphify-cache"


def test_cache_hit_across_worktrees(tmp_path):
    """A file cached from the main tree is reused when that file appears in a worktree."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    # Same relative path + same content, different absolute paths.
    main_file = repo / "src" / "foo.py"
    main_file.write_text("def greet(): return 'hello'\n")

    # Save cache from the main tree
    result = {"nodes": [{"id": "greet"}], "edges": []}
    save_cached(main_file, result, root=repo)

    # Now set up the worktree with the SAME file content at the SAME relative path
    wt_git = git_dir / "worktrees" / "feature"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feature\n")
    worktree = tmp_path / "worktrees" / "feature"
    (worktree / "src").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {wt_git}\n")
    wt_file = worktree / "src" / "foo.py"
    wt_file.write_text("def greet(): return 'hello'\n")

    # Loading from the worktree should hit the cache the main tree wrote.
    loaded = load_cached(wt_file, root=worktree)
    assert loaded == result


def test_file_hash_portable_across_worktrees(tmp_path):
    """Same content + same relative path → same hash, even at different absolute paths."""
    repo_main = tmp_path / "main"
    repo_wt = tmp_path / "worktree"
    (repo_main / "src").mkdir(parents=True)
    (repo_wt / "src").mkdir(parents=True)

    f_main = repo_main / "src" / "foo.py"
    f_wt = repo_wt / "src" / "foo.py"
    f_main.write_text("x = 1\n")
    f_wt.write_text("x = 1\n")

    assert file_hash(f_main, root=repo_main) == file_hash(f_wt, root=repo_wt)


def test_md_frontmatter_only_change_same_hash(tmp_path):
    """Changing only frontmatter fields in a .md file does not change the hash."""
    f = tmp_path / "doc.md"
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nBody text.")
    h1 = file_hash(f)
    f.write_text("---\nreviewed: 2026-04-09\n---\n\n# Title\n\nBody text.")
    h2 = file_hash(f)
    assert h1 == h2


def test_md_body_change_different_hash(tmp_path):
    """Changing the body of a .md file produces a different hash."""
    f = tmp_path / "doc.md"
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nOriginal body.")
    h1 = file_hash(f)
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nChanged body.")
    h2 = file_hash(f)
    assert h1 != h2


def test_md_no_frontmatter_hashed_normally(tmp_path):
    """A .md file with no frontmatter is hashed by its full content."""
    f = tmp_path / "doc.md"
    f.write_text("# Just a heading\n\nNo frontmatter here.")
    h1 = file_hash(f)
    f.write_text("# Just a heading\n\nDifferent content.")
    h2 = file_hash(f)
    assert h1 != h2


def test_non_md_file_hashed_fully(tmp_path):
    """Non-.md files are still hashed by their full content."""
    f = tmp_path / "script.py"
    f.write_text("# comment\nx = 1")
    h1 = file_hash(f)
    f.write_text("# changed comment\nx = 1")
    h2 = file_hash(f)
    assert h1 != h2


def test_body_content_strips_frontmatter():
    """_body_content correctly strips YAML frontmatter."""
    content = b"---\ntitle: Test\n---\n\nActual body."
    assert _body_content(content) == b"\n\nActual body."


def test_body_content_no_frontmatter():
    """_body_content returns content unchanged when no frontmatter present."""
    content = b"No frontmatter here."
    assert _body_content(content) == content
