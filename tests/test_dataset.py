import os
import pytest
import tempfile
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import ChimpanzeeDataset


def create_test_image(path, size=(112, 112)):
    """Create a test image at the specified path."""
    img = Image.fromarray(np.random.randint(0, 255, size, dtype=np.uint8))
    img.save(path)


def create_test_annotation_file(path, entries):
    """Create a test annotation file with the given entries."""
    with open(path, 'w') as f:
        for entry in entries:
            f.write(entry + '\n')


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with test images and annotations."""
    tmpdir = tempfile.TemporaryDirectory()
    root_dir = tmpdir.name
    
    # Create image directory
    image_dir = os.path.join(root_dir, 'images')
    os.makedirs(image_dir)
    
    # Create annotation entries
    # identity1: 25 samples (should pass filtering with min_samples=20)
    # identity2: 10 samples (should be filtered out with min_samples=20)
    # identity3: 30 samples (should pass filtering)
    annotation_entries = []
    
    for i in range(25):
        filename = f'identity1_male_{i:03d}.jpg'
        img_path = os.path.join(image_dir, filename)
        create_test_image(img_path)
        annotation_entries.append(
            f'{filename} 10 Adult 0 0 0 0 0 0'
        )
    
    for i in range(10):
        filename = f'identity2_female_{i:03d}.jpg'
        img_path = os.path.join(image_dir, filename)
        create_test_image(img_path)
        annotation_entries.append(
            f'{filename} 8 Adult 0 0 0 0 0 0'
        )
    
    for i in range(30):
        filename = f'identity3_male_{i:03d}.jpg'
        img_path = os.path.join(image_dir, filename)
        create_test_image(img_path)
        annotation_entries.append(
            f'{filename} 12 Adult 0 0 0 0 0 0'
        )
    
    # Create annotation file
    ann_file = os.path.join(root_dir, 'annotations.txt')
    create_test_annotation_file(ann_file, annotation_entries)
    
    yield {
        'root_dir': root_dir,
        'annotation_file': 'annotations.txt',
        'image_dir': 'images',
        'identities': ['identity1', 'identity3']  # Only these should pass filtering
    }
    
    tmpdir.cleanup()


class TestChimpanzeeDataset:
    def test_dataset_creation(self, mock_dataset):
        """Test basic dataset creation and sample count."""
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            min_samples_per_identity=20
        )
        
        # Should only include identity1 (25) and identity3 (30)
        assert len(dataset) == 55
        assert len(dataset.identities) == 2
        assert 'identity1' in dataset.identities
        assert 'identity3' in dataset.identities
        assert 'identity2' not in dataset.identities
    
    def test_dataset_filtering(self, mock_dataset):
        """Test filtering of identities with insufficient samples."""
        # With min_samples=5, all three identities should be included
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            min_samples_per_identity=5
        )
        
        assert len(dataset.identities) == 3
        assert len(dataset) == 65  # 25 + 10 + 30
        
        # With min_samples=20, only identity1 and identity3 should be included
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            min_samples_per_identity=20
        )
        
        assert len(dataset.identities) == 2
        assert len(dataset) == 55  # 25 + 30
    
    def test_dataset_getitem(self, mock_dataset):
        """Test __getitem__ returns correct format."""
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            min_samples_per_identity=20
        )
        
        # Get first sample
        image, label = dataset[0]
        
        # Check image is PIL Image
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        
        # Check label is integer
        assert isinstance(label, int)
        assert label >= 0
        assert label < len(dataset.identities)
        
        # Check we can access all samples
        for idx in range(len(dataset)):
            img, lbl = dataset[idx]
            assert isinstance(img, Image.Image)
            assert isinstance(lbl, int)
    
    def test_dataset_with_specified_identities(self, mock_dataset):
        """Test dataset with explicitly specified identities."""
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            identities=['identity1']
        )
        
        assert len(dataset.identities) == 1
        assert len(dataset) == 25
        assert dataset.identities[0] == 'identity1'
    
    def test_dataset_with_transform(self, mock_dataset):
        """Test that transforms are applied."""
        # Create a simple transform that converts to numpy array
        def simple_transform(img):
            return np.array(img)
        
        dataset = ChimpanzeeDataset(
            root_dir=mock_dataset['root_dir'],
            annotation_file=mock_dataset['annotation_file'],
            image_dir=mock_dataset['image_dir'],
            min_samples_per_identity=20,
            transform=simple_transform
        )
        
        image, label = dataset[0]
        
        # After transform, should be numpy array not PIL Image
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # RGB channels
