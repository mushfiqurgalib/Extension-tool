import os
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
import subprocess
from urllib.parse import urlparse

# Configuration
BASE_DIR = "f:/Thesis_methodology"
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_FILE = os.environ.get(
    "TARGET_FILE",
    os.path.join(BASE_DIR, "minGPT", "mingpt", "model.py")
)


def find_repo_root(start_path):
    current = os.path.abspath(os.path.dirname(start_path))
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(os.path.dirname(start_path))
        current = parent


def extract_repo_info(target_file):
    repo_root = find_repo_root(target_file)
    repo_name = os.path.basename(repo_root)
    repo_owner = None

    try:
        remote_url = subprocess.check_output(
            ["git", "-C", repo_root, "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        if remote_url.startswith("git@"):
            path_part = remote_url.split(":", 1)[1]
        else:
            path_part = urlparse(remote_url).path.lstrip("/")

        path_parts = path_part.removesuffix(".git").split("/")
        if len(path_parts) >= 2:
            repo_owner = path_parts[-2]
            repo_name = path_parts[-1]
    except Exception:
        pass

    return repo_owner, repo_name, repo_root

def get_installed_version(package):
    try:
        output = subprocess.check_output(["pip", "show", package]).decode()
        for line in output.split("\n"):
            if line.startswith("Version:"):
                return line.split(":")[1].strip()
    except:
        return "latest"

def scrape_github_artifacts(owner, repo):
    repo_label = f"{owner}/{repo}" if owner else repo
    print(f"Scraping GitHub artifacts for {repo_label}...")
    artifacts = []

    _, _, repo_root = extract_repo_info(TARGET_FILE)

    # 1. README
    readme_path = os.path.join(repo_root, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            artifacts.append({"content": f.read(), "source": "README"})
            print("Found README.md")

    # 2. Issues and PRs (Public API)
    if owner:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=closed&per_page=10"
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                issues = response.json()
                for issue in issues:
                    content = f"Title: {issue['title']}\nBody: {issue.get('body', '')}"
                    artifacts.append({"content": content, "source": f"Issue/PR {issue['number']}"})
                print(f"Fetched {len(issues)} closed issues/PRs")
            else:
                print(f"Failed to fetch issues: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching GitHub artifacts: {e}")
    else:
        print("No GitHub remote owner found. Skipping issue/PR fetch.")

    return artifacts

def extract_pypi_docs(package, version):
  
    """Extract the PyPI project description for a specific package version.
    
    This function builds the canonical PyPI project URL for ``package`` and
    ``version``, downloads the corresponding project page, and parses the HTML
    to locate the element with ``id="description"``. If found, the description
    text is returned as a single artifact dictionary in a list.
    
    Parameters
    ----------
    package : str
        Name of the PyPI package.
    version : str
        Version string of the PyPI release to fetch.
    
    Returns
    -------
    list of dict
        A list containing zero or one artifact dictionaries. Each artifact has
        the keys ``"content"`` and ``"source"``. The list is empty if the page
        cannot be retrieved or no description element is present.
    
    Notes
    -----
    This routine relies on the standard PyPI project page layout and uses
    ``requests`` for HTTP retrieval and ``BeautifulSoup`` from ``bs4`` for HTML
    parsing. It is designed to extract documentation text from the package's
    published PyPI page, not from local source distributions or Sphinx builds.
    """
    print(f"Extracting PyPI docs for {package} v{version}...")
    url = f"https://pypi.org/project/{package}/{version}/"
    artifacts = []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            description = soup.find(id="description")
            if description:
                artifacts.append({"content": description.get_text(), "source": f"PyPI {package} v{version}"})
                print("Extracted PyPI project description.")
        else:
             print(f"PyPI page not found: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching PyPI docs: {e}")
    return artifacts

def build_vector_db(artifacts):
        """Build a persistent ChromaDB vector store from artifact contents.
    
    Initializes a persistent ChromaDB client and a ``repo_context`` collection,
    loads the configured SentenceTransformer embedding model, and splits each
    artifact's ``content`` field into overlapping character chunks. Each chunk is
    paired with the originating artifact's ``source`` metadata and a unique
    incremental ID, preparing the data for later embedding and insertion into the
    vector database. If no chunks are produced, the function exits early.
    
    Parameters
    ----------
    artifacts : list of dict
        Artifact records to index. Each record must provide:
    
        - ``content`` : str
            Text to be chunked and embedded.
        - ``source`` : str
            Source identifier stored as chunk metadata.
    
    Notes
    -----
    This function is part of a retrieval-oriented preprocessing pipeline for
    building repository context. It uses ChromaDB's persistent client API and a
    SentenceTransformer embedding model, with chunking performed by
    ``CharacterTextSplitter(chunk_size=500, chunk_overlap=50)``.
    
    See Also
    --------
    chromadb.PersistentClient : Persistent vector database client.
    SentenceTransformer : Sentence embedding model loader.
    CharacterTextSplitter : Text splitter for overlapping chunk generation.
    """
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="repo_context")
    
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    idx = 0
    for art in artifacts:
        chunks = text_splitter.split_text(art['content'])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": art['source']})
            all_ids.append(f"id_{idx}")
            idx += 1
            
    if not all_chunks:
        print("No chunks to add to Vector DB.")
        return

    print(f"Encoding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks).tolist()
    
    print("Adding to collection...")
    # Add in batches to avoid size limits
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            embeddings=embeddings[i:end],
            documents=all_chunks[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end]
        )
    print(f"Successfully added {len(all_chunks)} chunks to ChromaDB at {DB_PATH}")

if __name__ == "__main__":
    repo_owner, repo_name, _ = extract_repo_info(TARGET_FILE)
    github_arts = scrape_github_artifacts(repo_owner, repo_name)
    torch_version = get_installed_version("torch")
    pypi_arts = extract_pypi_docs("torch", torch_version)

    all_artifacts = github_arts + pypi_arts
    build_vector_db(all_artifacts)
