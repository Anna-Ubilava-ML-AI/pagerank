import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    distribution = {}
    N_pages = len(corpus)
    random_prob = (1 - damping_factor) / N_pages
    
    # Random probability for all pages
    for p in corpus:
        distribution[p] = random_prob
    
    # If page has outgoing links(NumLinks), distribute damping probability
    numlinks = corpus[page]
    if numlinks:
        link_prob = damping_factor / len(numlinks)
        for link in numlinks:
            distribution[link] += link_prob
    # If no outgoing links, distribute randomly among all pages with equal probability. 
    else:
        additional_prob = damping_factor / N_pages
        for p in corpus:
            distribution[p] += additional_prob
    
    return distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_counts = {page: 0 for page in corpus}
    
    # generate first sample by choosing from a page at random
    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1
    
    # generate the remaining samples using the transition model of the previous sample
    for _ in range(n - 1):
        probabilities = transition_model(corpus, current_page, damping_factor)
        # convert probabilities to a list of pages and weights for random.choices
        pages = list(probabilities.keys())
        weights = list(probabilities.values())
        current_page = random.choices(pages, weights=weights, k=1)[0]
        page_counts[current_page] += 1
    
    # convert counts to probabilities
    return {page: count / n for page, count in page_counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N_pages = len(corpus)
    
    # a dictionary to track pages that link to each page
    inbound_links = {page: set() for page in corpus}
    for page, links in corpus.items():
        for link in links:
            inbound_links[link].add(page)
    
    # handel pages with no outgoing links
    for page, links in corpus.items():
        if not links:
            # If a page has no links, we treat it as having links to all pages
            corpus[page] = set(corpus.keys())
    
    # Start with assigning each page a rank of 1 / N (equal probability for all pages)
    page_rank = {page: 1 / N_pages for page in corpus}
    
    # Iteratively calculate new ranks until convergence (based on all the previous set of PageRank values,)
    while True:
        new_rank = {}
        max_change = 0
        
        for page in corpus:
            # calculate the contribution from all pages that link to this page
            link_sum = 0
            for linking_page in inbound_links[page]:
                link_sum += page_rank[linking_page] / len(corpus[linking_page])
            
            # Cclculate new rank using the PageRank formula
            new_rank[page] = (1 - damping_factor) / N_pages + damping_factor * link_sum
            
            # track the largest change in rank
            change = abs(new_rank[page] - page_rank[page])
            max_change = max(max_change, change)
        
        # update page ranks
        page_rank = new_rank
        
        # check for convergence (max value changes can be 0.001)
        if max_change < 0.001:
            break
    
    # normalization (to ensure the sum of all PageRanks is 1)
    total = sum(page_rank.values())
    return {page: rank / total for page, rank in page_rank.items()}


if __name__ == "__main__":
    main()
