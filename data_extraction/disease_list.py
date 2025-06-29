import xml.etree.ElementTree as ET

tree = ET.parse('./data/mesh/desc2025.xml')
root = tree.getroot()

disease_names = set()

# Extract disease terms using TreeNumber codes starting with 'C'
for descriptor in root.findall('.//DescriptorRecord'):
    tree_numbers = descriptor.findall('./TreeNumberList/TreeNumber')
    if any(tree_number.text.startswith('C') for tree_number in tree_numbers):
        name_element = descriptor.find('./DescriptorName/String')
        if name_element is not None:
            disease_names.add(name_element.text)

DISEASES = sorted(disease_names)
