"""
Git Integration Module.

Provides Git version control integration for strategy management.
Supports automatic commits, tagging, history tracking, and branch management.
"""

import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .versioning import (
    VersionManager,
    StrategyMetadata,
    StrategySnapshot,
    SerializationFormat,
)


class GitError(Exception):
    """Exception raised when Git operation fails."""
    pass


class GitOperationType(str, Enum):
    """Git operation types for logging."""
    INIT = "init"
    COMMIT = "commit"
    TAG = "tag"
    BRANCH = "branch"
    CHECKOUT = "checkout"
    MERGE = "merge"
    DIFF = "diff"
    LOG = "log"
    PUSH = "push"
    PULL = "pull"


@dataclass
class GitCommit:
    """
    Git commit information.

    Attributes:
        hash: Commit hash (full)
        short_hash: Short commit hash
        author: Author name
        email: Author email
        date: Commit date
        message: Commit message
        files_changed: List of changed files
    """
    hash: str
    short_hash: str
    author: str
    email: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)

    @classmethod
    def from_log_line(cls, log_output: str) -> "GitCommit":
        """Parse from git log --format output."""
        # Expected format: hash|short|author|email|date|message
        parts = log_output.strip().split("|")
        if len(parts) < 6:
            raise ValueError(f"Invalid log format: {log_output}")

        return cls(
            hash=parts[0],
            short_hash=parts[1],
            author=parts[2],
            email=parts[3],
            date=datetime.fromisoformat(parts[4].replace(" ", "T")),
            message="|".join(parts[5:]),  # Message might contain |
        )


@dataclass
class GitTag:
    """
    Git tag information.

    Attributes:
        name: Tag name
        commit_hash: Associated commit hash
        message: Tag message (for annotated tags)
        date: Tag date
    """
    name: str
    commit_hash: str
    message: str = ""
    date: Optional[datetime] = None


@dataclass
class GitBranch:
    """
    Git branch information.

    Attributes:
        name: Branch name
        is_current: Whether this is the current branch
        commit_hash: HEAD commit hash
        tracking: Remote tracking branch (if any)
    """
    name: str
    is_current: bool = False
    commit_hash: str = ""
    tracking: Optional[str] = None


@dataclass
class GitStatus:
    """
    Git repository status.

    Attributes:
        is_clean: Whether working directory is clean
        branch: Current branch name
        staged: List of staged files
        modified: List of modified files
        untracked: List of untracked files
        ahead: Commits ahead of remote
        behind: Commits behind remote
    """
    is_clean: bool = True
    branch: str = "main"
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0


class GitClient:
    """
    Low-level Git client for executing Git commands.

    Wraps subprocess calls to Git CLI with error handling.
    """

    def __init__(self, repo_path: Path):
        """
        Initialize Git client.

        Args:
            repo_path: Path to Git repository
        """
        self._repo_path = repo_path

    def _run(
        self,
        args: list[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a Git command.

        Args:
            args: Git command arguments (without 'git')
            check: Raise exception on non-zero exit
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        cmd = ["git", "-C", str(self._repo_path)] + args

        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=capture_output,
                text=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {' '.join(args)}\n{e.stderr}")
        except FileNotFoundError:
            raise GitError("Git is not installed or not in PATH")

    def is_repo(self) -> bool:
        """Check if path is a Git repository."""
        try:
            result = self._run(["rev-parse", "--git-dir"], check=False)
            return result.returncode == 0
        except GitError:
            return False

    def init(self, initial_branch: str = "main") -> None:
        """Initialize a new Git repository."""
        self._run(["init", "-b", initial_branch])

    def config(self, key: str, value: str, is_global: bool = False) -> None:
        """
        Set a Git configuration value.

        Args:
            key: Configuration key (e.g., "user.name")
            value: Configuration value
            is_global: Whether to set globally
        """
        args = ["config"]
        if is_global:
            args.append("--global")
        args.extend([key, value])
        self._run(args)

    def get_config(self, key: str) -> Optional[str]:
        """
        Get a Git configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value or None if not set
        """
        try:
            result = self._run(["config", "--get", key], check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except GitError:
            pass
        return None

    def add(self, paths: list[str]) -> None:
        """Stage files for commit."""
        if paths:
            self._run(["add"] + paths)

    def commit(self, message: str, allow_empty: bool = False) -> str:
        """
        Create a commit.

        Args:
            message: Commit message
            allow_empty: Allow empty commits

        Returns:
            Commit hash
        """
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")

        self._run(args)

        # Get commit hash
        result = self._run(["rev-parse", "HEAD"])
        return result.stdout.strip()

    def tag(
        self,
        name: str,
        message: Optional[str] = None,
        commit: Optional[str] = None,
    ) -> None:
        """
        Create a tag.

        Args:
            name: Tag name
            message: Tag message (creates annotated tag if provided)
            commit: Commit to tag (defaults to HEAD)
        """
        args = ["tag"]
        if message:
            args.extend(["-a", name, "-m", message])
        else:
            args.append(name)

        if commit:
            args.append(commit)

        self._run(args)

    def get_tags(self) -> list[GitTag]:
        """Get all tags."""
        result = self._run(["tag", "-l"])
        tags = []

        for tag_name in result.stdout.strip().split("\n"):
            if not tag_name:
                continue

            # Get commit hash for tag
            hash_result = self._run(["rev-list", "-n", "1", tag_name])
            commit_hash = hash_result.stdout.strip()

            # Try to get tag message
            message = ""
            try:
                msg_result = self._run(["tag", "-l", "-n1", tag_name])
                parts = msg_result.stdout.strip().split(None, 1)
                if len(parts) > 1:
                    message = parts[1]
            except GitError:
                pass

            tags.append(GitTag(
                name=tag_name,
                commit_hash=commit_hash,
                message=message,
            ))

        return tags

    def branch(self, name: str, start_point: Optional[str] = None) -> None:
        """Create a new branch."""
        args = ["branch", name]
        if start_point:
            args.append(start_point)
        self._run(args)

    def checkout(self, ref: str, create: bool = False) -> None:
        """
        Checkout a branch or commit.

        Args:
            ref: Branch name, tag, or commit hash
            create: Create branch if it doesn't exist
        """
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(ref)
        self._run(args)

    def get_branches(self) -> list[GitBranch]:
        """Get all branches."""
        result = self._run(["branch", "-vv"])
        branches = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            is_current = line.startswith("*")
            line = line[2:].strip()  # Remove leading "* " or "  "

            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                commit_hash = parts[1]

                # Check for tracking info
                tracking = None
                if "[" in line:
                    match = re.search(r"\[([^\]]+)\]", line)
                    if match:
                        tracking = match.group(1).split(":")[0]

                branches.append(GitBranch(
                    name=name,
                    is_current=is_current,
                    commit_hash=commit_hash,
                    tracking=tracking,
                ))

        return branches

    def get_current_branch(self) -> str:
        """Get current branch name."""
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def status(self) -> GitStatus:
        """Get repository status."""
        result = self._run(["status", "--porcelain", "-b"])
        lines = result.stdout.strip().split("\n")

        status = GitStatus()
        staged = []
        modified = []
        untracked = []

        for line in lines:
            if not line:
                continue

            if line.startswith("##"):
                # Branch line
                branch_info = line[3:]
                if "..." in branch_info:
                    status.branch = branch_info.split("...")[0]
                    # Check ahead/behind
                    if "[" in branch_info:
                        match = re.search(r"\[ahead (\d+)", branch_info)
                        if match:
                            status.ahead = int(match.group(1))
                        match = re.search(r"behind (\d+)", branch_info)
                        if match:
                            status.behind = int(match.group(1))
                else:
                    status.branch = branch_info
            else:
                # File status
                index_status = line[0]
                worktree_status = line[1]
                filename = line[3:]

                if index_status in "MADRCU":
                    staged.append(filename)
                if worktree_status in "MD":
                    modified.append(filename)
                if index_status == "?" and worktree_status == "?":
                    untracked.append(filename)

        status.staged = staged
        status.modified = modified
        status.untracked = untracked
        status.is_clean = not (staged or modified or untracked)

        return status

    def log(
        self,
        n: int = 10,
        path: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[GitCommit]:
        """
        Get commit history.

        Args:
            n: Number of commits to retrieve
            path: Filter by file path
            since: Filter commits after this date

        Returns:
            List of GitCommit objects
        """
        format_str = "%H|%h|%an|%ae|%ci|%s"
        args = ["log", f"-n{n}", f"--format={format_str}"]

        if since:
            args.append(f"--since={since.isoformat()}")

        if path:
            args.extend(["--", path])

        result = self._run(args)
        commits = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                commits.append(GitCommit.from_log_line(line))
            except ValueError:
                continue

        return commits

    def diff(
        self,
        ref1: str,
        ref2: Optional[str] = None,
        path: Optional[str] = None,
        stat_only: bool = False,
    ) -> str:
        """
        Get diff between refs.

        Args:
            ref1: First reference (commit, branch, tag)
            ref2: Second reference (defaults to working tree)
            path: Filter by file path
            stat_only: Only show stat summary

        Returns:
            Diff output
        """
        args = ["diff"]
        if stat_only:
            args.append("--stat")

        args.append(ref1)
        if ref2:
            args.append(ref2)

        if path:
            args.extend(["--", path])

        result = self._run(args)
        return result.stdout

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        tags: bool = False,
        force: bool = False,
    ) -> None:
        """
        Push to remote.

        Args:
            remote: Remote name
            branch: Branch to push (defaults to current)
            tags: Push tags
            force: Force push (use with caution)
        """
        args = ["push", remote]

        if branch:
            args.append(branch)

        if tags:
            args.append("--tags")

        if force:
            args.append("--force")

        self._run(args)

    def pull(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        rebase: bool = False,
    ) -> None:
        """
        Pull from remote.

        Args:
            remote: Remote name
            branch: Branch to pull (defaults to current)
            rebase: Use rebase instead of merge
        """
        args = ["pull"]

        if rebase:
            args.append("--rebase")

        args.append(remote)

        if branch:
            args.append(branch)

        self._run(args)

    def get_file_at_commit(self, commit: str, path: str) -> str:
        """Get file contents at a specific commit."""
        result = self._run(["show", f"{commit}:{path}"])
        return result.stdout


class GitVersionManager(VersionManager):
    """
    Git-integrated version manager.

    Extends VersionManager with Git operations for automatic
    commits, tagging, and history tracking.

    Example:
        manager = GitVersionManager(
            base_path=Path("./strategies"),
            auto_commit=True,
            auto_tag=True,
        )

        # Save version (automatically commits and tags)
        snapshot = manager.save_version(config, metadata)

        # Get commit history for a strategy
        history = manager.get_strategy_history("my_strategy")

        # Compare versions using Git diff
        diff = manager.diff_versions("my_strategy", "1.0.0", "1.1.0")

        # Create a branch for experimentation
        manager.create_experiment_branch("my_strategy", "new_feature")
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        format: SerializationFormat = SerializationFormat.JSON,
        auto_commit: bool = True,
        auto_tag: bool = True,
        auto_push: bool = False,
        remote: str = "origin",
    ):
        """
        Initialize Git version manager.

        Args:
            base_path: Base directory for storing versions
            format: Serialization format to use
            auto_commit: Automatically commit on save
            auto_tag: Automatically create tags for versions
            auto_push: Automatically push to remote
            remote: Remote name for push operations
        """
        super().__init__(base_path, format)

        self._auto_commit = auto_commit
        self._auto_tag = auto_tag
        self._auto_push = auto_push
        self._remote = remote

        self._git = GitClient(self._base_path)

        # Initialize repo if needed
        if not self._git.is_repo():
            self._init_repo()

    def _init_repo(self) -> None:
        """Initialize Git repository with initial structure."""
        self._git.init()

        # Ensure user config is set for commits
        if not self._git.get_config("user.name"):
            self._git.config("user.name", "Strategy Version Control")
        if not self._git.get_config("user.email"):
            self._git.config("user.email", "strategy@localhost")

        # Create .gitignore
        gitignore_path = self._base_path / ".gitignore"
        gitignore_content = """# Strategy version control
*.pyc
__pycache__/
.DS_Store
*.log
.env
"""
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)

        # Create README
        readme_path = self._base_path / "README.md"
        readme_content = """# Strategy Versions

This directory contains versioned trading strategy configurations.

## Structure

```
strategies/
├── strategy_name/
│   ├── v1.0.0.json
│   ├── v1.1.0.json
│   └── latest.txt
└── ...
```

## Usage

Managed by GitVersionManager.
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Initial commit
        self._git.add([".gitignore", "README.md"])
        self._git.commit("Initial repository setup")

    def save_version(
        self,
        config: Any,
        metadata: StrategyMetadata,
        commit_message: Optional[str] = None,
    ) -> StrategySnapshot:
        """
        Save a strategy version with Git integration.

        Args:
            config: Strategy configuration
            metadata: Strategy metadata
            commit_message: Custom commit message (auto-generated if None)

        Returns:
            StrategySnapshot of the saved version
        """
        # Call parent save_version
        snapshot = super().save_version(config, metadata)

        # Git operations
        if self._auto_commit:
            # Stage the version file
            version_file = self._get_version_file(metadata.name, metadata.version)
            latest_file = self._get_strategy_dir(metadata.name) / "latest.txt"

            relative_version = version_file.relative_to(self._base_path)
            relative_latest = latest_file.relative_to(self._base_path)

            self._git.add([str(relative_version), str(relative_latest)])

            # Commit
            if commit_message is None:
                commit_message = self._generate_commit_message(metadata)

            commit_hash = self._git.commit(commit_message)

            # Tag
            if self._auto_tag:
                tag_name = f"{metadata.name}/v{metadata.version}"
                tag_message = f"Strategy: {metadata.name}\nVersion: {metadata.version}\n\n{metadata.notes}"
                self._git.tag(tag_name, tag_message)

            # Push
            if self._auto_push:
                try:
                    self._git.push(self._remote, tags=self._auto_tag)
                except GitError:
                    # Push failure shouldn't fail the save
                    pass

        return snapshot

    def _generate_commit_message(self, metadata: StrategyMetadata) -> str:
        """Generate commit message from metadata."""
        lines = [
            f"Update {metadata.name} to v{metadata.version}",
            "",
        ]

        if metadata.description:
            lines.append(metadata.description)
            lines.append("")

        if metadata.notes:
            lines.append("Notes:")
            lines.append(metadata.notes)
            lines.append("")

        if metadata.tags:
            lines.append(f"Tags: {', '.join(metadata.tags)}")

        return "\n".join(lines)

    def get_strategy_history(
        self,
        strategy_name: str,
        n: int = 20,
    ) -> list[GitCommit]:
        """
        Get commit history for a strategy.

        Args:
            strategy_name: Name of the strategy
            n: Number of commits to retrieve

        Returns:
            List of commits affecting this strategy
        """
        strategy_dir = self._get_strategy_dir(strategy_name)
        relative_path = strategy_dir.relative_to(self._base_path)

        return self._git.log(n=n, path=str(relative_path))

    def get_all_tags(self, strategy_name: Optional[str] = None) -> list[GitTag]:
        """
        Get all version tags.

        Args:
            strategy_name: Filter by strategy name (optional)

        Returns:
            List of tags
        """
        tags = self._git.get_tags()

        if strategy_name:
            prefix = f"{strategy_name}/v"
            tags = [t for t in tags if t.name.startswith(prefix)]

        return tags

    def diff_versions(
        self,
        strategy_name: str,
        version1: str,
        version2: str,
    ) -> dict[str, Any]:
        """
        Get diff between two versions using Git.

        Args:
            strategy_name: Name of the strategy
            version1: First version
            version2: Second version

        Returns:
            Dictionary with diff information
        """
        tag1 = f"{strategy_name}/v{version1}"
        tag2 = f"{strategy_name}/v{version2}"

        # Get stat diff
        stat_diff = self._git.diff(tag1, tag2, stat_only=True)

        # Get full diff for the version file
        ext = "json" if self._format == SerializationFormat.JSON else "yaml"
        version_file = f"{strategy_name}/v{version2}.{ext}"
        full_diff = self._git.diff(tag1, tag2, path=version_file)

        # Also get the structured comparison
        structured_diff = self.compare_versions(strategy_name, version1, version2)

        return {
            "version1": version1,
            "version2": version2,
            "tag1": tag1,
            "tag2": tag2,
            "stat": stat_diff,
            "diff": full_diff,
            "structured": structured_diff,
        }

    def get_version_at_commit(
        self,
        strategy_name: str,
        version: str,
        commit: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get strategy configuration at a specific commit.

        Args:
            strategy_name: Name of the strategy
            version: Version string
            commit: Commit hash

        Returns:
            Configuration dict or None if not found
        """
        ext = "json" if self._format == SerializationFormat.JSON else "yaml"
        file_path = f"{strategy_name}/v{version}.{ext}"

        try:
            content = self._git.get_file_at_commit(commit, file_path)

            if self._format == SerializationFormat.JSON:
                import json
                return json.loads(content)
            else:
                import yaml
                return yaml.safe_load(content)
        except GitError:
            return None

    def create_experiment_branch(
        self,
        strategy_name: str,
        experiment_name: str,
        from_version: Optional[str] = None,
    ) -> str:
        """
        Create a branch for strategy experimentation.

        Args:
            strategy_name: Name of the strategy
            experiment_name: Name of the experiment
            from_version: Version to branch from (latest if None)

        Returns:
            Branch name
        """
        branch_name = f"experiment/{strategy_name}/{experiment_name}"

        # Get starting point
        start_point = None
        if from_version:
            start_point = f"{strategy_name}/v{from_version}"

        self._git.branch(branch_name, start_point)
        return branch_name

    def checkout_experiment(self, branch_name: str) -> None:
        """Checkout an experiment branch."""
        self._git.checkout(branch_name)

    def merge_experiment(
        self,
        branch_name: str,
        target_branch: str = "main",
    ) -> None:
        """
        Merge experiment branch into target.

        Args:
            branch_name: Experiment branch to merge
            target_branch: Target branch (default: main)
        """
        current = self._git.get_current_branch()

        # Checkout target
        self._git.checkout(target_branch)

        # Merge
        self._git._run(["merge", branch_name, "--no-ff", "-m", f"Merge {branch_name}"])

        # Return to original branch
        if current != target_branch:
            self._git.checkout(current)

    def get_status(self) -> GitStatus:
        """Get repository status."""
        return self._git.status()

    def get_branches(self) -> list[GitBranch]:
        """Get all branches."""
        return self._git.get_branches()

    def sync_with_remote(self, push: bool = True, pull: bool = True) -> dict[str, bool]:
        """
        Sync with remote repository.

        Args:
            push: Push local changes
            pull: Pull remote changes

        Returns:
            Dict with operation results
        """
        results = {"pull": False, "push": False}

        if pull:
            try:
                self._git.pull(self._remote)
                results["pull"] = True
            except GitError:
                pass

        if push:
            try:
                self._git.push(self._remote, tags=True)
                results["push"] = True
            except GitError:
                pass

        return results

    def rollback_to_version(
        self,
        strategy_name: str,
        version: str,
        create_new_version: bool = True,
    ) -> Optional[StrategySnapshot]:
        """
        Rollback a strategy to a previous version.

        Args:
            strategy_name: Name of the strategy
            version: Version to rollback to
            create_new_version: If True, creates a new version with rollback content
                              If False, just checks out the old version file

        Returns:
            New snapshot if create_new_version is True
        """
        # Load the old version
        config, metadata = self.load_version(strategy_name, version)

        if create_new_version:
            # Get current latest version
            versions = self.list_versions(strategy_name)
            if versions:
                current = versions[0]
                # Increment patch version
                parts = current.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                new_version = ".".join(parts)
            else:
                new_version = "1.0.0"

            # Create new metadata
            new_metadata = StrategyMetadata(
                name=strategy_name,
                version=new_version,
                description=metadata.description,
                author=metadata.author,
                tags=metadata.tags,
                parent_version=version,
                notes=f"Rollback to v{version}",
            )

            # Save as new version
            return self.save_version(
                config,
                new_metadata,
                commit_message=f"Rollback {strategy_name} to v{version}",
            )

        return None

    def export_history_report(
        self,
        strategy_name: str,
        format: str = "text",
    ) -> str:
        """
        Generate a history report for a strategy.

        Args:
            strategy_name: Name of the strategy
            format: Output format ('text' or 'markdown')

        Returns:
            Formatted report string
        """
        versions = self.list_versions(strategy_name)
        tags = self.get_all_tags(strategy_name)
        history = self.get_strategy_history(strategy_name)

        if format == "markdown":
            return self._generate_markdown_history(strategy_name, versions, tags, history)
        return self._generate_text_history(strategy_name, versions, tags, history)

    def _generate_text_history(
        self,
        strategy_name: str,
        versions: list[str],
        tags: list[GitTag],
        history: list[GitCommit],
    ) -> str:
        """Generate text history report."""
        lines = [
            "=" * 60,
            f"STRATEGY HISTORY: {strategy_name}",
            "=" * 60,
            "",
            f"Total Versions: {len(versions)}",
            f"Total Commits: {len(history)}",
            "",
            "VERSIONS:",
            "-" * 40,
        ]

        for version in versions:
            tag = next((t for t in tags if t.name == f"{strategy_name}/v{version}"), None)
            tag_info = f" (tagged: {tag.name})" if tag else ""
            lines.append(f"  v{version}{tag_info}")

        lines.extend([
            "",
            "RECENT COMMITS:",
            "-" * 40,
        ])

        for commit in history[:10]:
            date_str = commit.date.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{commit.short_hash}] {date_str} - {commit.message}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _generate_markdown_history(
        self,
        strategy_name: str,
        versions: list[str],
        tags: list[GitTag],
        history: list[GitCommit],
    ) -> str:
        """Generate markdown history report."""
        lines = [
            f"# Strategy History: {strategy_name}",
            "",
            f"- **Total Versions**: {len(versions)}",
            f"- **Total Commits**: {len(history)}",
            "",
            "## Versions",
            "",
            "| Version | Tag | Notes |",
            "|---------|-----|-------|",
        ]

        for version in versions:
            tag = next((t for t in tags if t.name == f"{strategy_name}/v{version}"), None)
            tag_name = tag.name if tag else "-"
            tag_msg = tag.message[:50] if tag and tag.message else "-"
            lines.append(f"| v{version} | {tag_name} | {tag_msg} |")

        lines.extend([
            "",
            "## Recent Commits",
            "",
        ])

        for commit in history[:10]:
            date_str = commit.date.strftime("%Y-%m-%d %H:%M")
            lines.append(f"- **{commit.short_hash}** ({date_str}): {commit.message}")

        return "\n".join(lines)


def create_git_version_manager(
    base_path: Optional[Path] = None,
    auto_commit: bool = True,
    auto_tag: bool = True,
    auto_push: bool = False,
) -> GitVersionManager:
    """
    Factory function to create GitVersionManager.

    Args:
        base_path: Base directory for storing versions
        auto_commit: Automatically commit on save
        auto_tag: Automatically create tags
        auto_push: Automatically push to remote

    Returns:
        Configured GitVersionManager
    """
    return GitVersionManager(
        base_path=base_path,
        auto_commit=auto_commit,
        auto_tag=auto_tag,
        auto_push=auto_push,
    )
