"""
Quick test for v1.5.0 core components.
Tests ProjectManager, VectorStore, and StateManager multi-tenancy.
"""

import sys
sys.path.insert(0, '/app')

from analyzer.project_manager import ProjectManager
from analyzer.state import StateManager
from analyzer.vector_store import VectorStore


def test_project_manager():
    """Test project management."""
    print("\n=== Testing ProjectManager ===")

    pm = ProjectManager(db_path='/tmp/test_projects.db')

    # Test: Create project
    project = pm.create_project(
        slug='test-case-123',
        name='Test Case 123',
        description='Test project',
        color='#ff0000'
    )
    print(f"✓ Created project: {project['name']} ({project['slug']})")

    # Test: List projects
    projects = pm.list_projects()
    print(f"✓ Listed {len(projects)} projects")
    assert len(projects) >= 2  # default + test-case-123

    # Test: Get project
    retrieved = pm.get_project('test-case-123')
    assert retrieved['name'] == 'Test Case 123'
    print(f"✓ Retrieved project: {retrieved['name']}")

    # Test: Update project
    updated = pm.update_project('test-case-123', description='Updated description')
    assert updated['description'] == 'Updated description'
    print(f"✓ Updated project description")

    # Test: Slug generation
    slug = pm.suggest_slug('Case 2024-456: Smith vs Jones')
    print(f"✓ Generated slug: {slug}")
    assert 'smith' in slug.lower()

    # Test: Delete project
    pm.delete_project('test-case-123', delete_data=False)
    print(f"✓ Deleted project")

    print("✅ ProjectManager tests passed\n")


def test_state_manager():
    """Test per-project state management."""
    print("\n=== Testing StateManager ===")

    # Test: Default project state
    state1 = StateManager(state_dir='/tmp/test_state', project_slug='default')
    print(f"✓ Created StateManager for 'default' project")
    print(f"  State file: {state1.state_path}")

    # Test: Another project state
    state2 = StateManager(state_dir='/tmp/test_state', project_slug='case-123')
    print(f"✓ Created StateManager for 'case-123' project")
    print(f"  State file: {state2.state_path}")

    # Verify separate state files
    assert state1.state_path != state2.state_path
    assert 'state_default' in str(state1.state_path)
    assert 'state_case-123' in str(state2.state_path)
    print(f"✓ State files are separate per project")

    print("✅ StateManager tests passed\n")


def test_vector_store():
    """Test multi-collection vector store."""
    print("\n=== Testing VectorStore ===")

    # Note: VectorStore requires Cohere API key
    # This test just verifies initialization works with project slugs

    # Test: Default project
    vs1 = VectorStore(project_slug='default', persist_directory='/tmp/test_chroma')
    print(f"✓ Created VectorStore for 'default' project")
    print(f"  Collection name: {vs1.collection_name}")
    assert vs1.collection_name == 'paperless_docs_default'

    # Test: Another project
    vs2 = VectorStore(project_slug='case-456', persist_directory='/tmp/test_chroma')
    print(f"✓ Created VectorStore for 'case-456' project")
    print(f"  Collection name: {vs2.collection_name}")
    assert vs2.collection_name == 'paperless_docs_case-456'

    # Verify different collections
    assert vs1.collection_name != vs2.collection_name
    print(f"✓ Collections are separate per project")

    print("✅ VectorStore tests passed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("v1.5.0 Core Components Test")
    print("=" * 60)

    try:
        test_project_manager()
        test_state_manager()
        test_vector_store()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
