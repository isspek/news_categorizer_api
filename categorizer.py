from ktrain import load_predictor
from loguru import logger

categorizer_model = './model/model'

expected_domains = {
    'politics': ['politics'],
    'sports': ['sports', 'sport', 'basketball', 'tennis', 'football'],
    'crime': ['crime', 'law'],
    'economy': ['economy', 'finance', 'business', 'markets', 'market', 'forbes'],
    'entertainment': ['entertainment', 'music', 'film', 'movie', 'hollywood', 'celebrity', 'celebrity-news'],
    'life': ['lifestyle', 'life-style', 'life', 'health'],
    'science and technology': ['technology', 'product', 'science', 'tech'],
    'uncategorized': ['uncategorized'],
}


model = load_predictor(categorizer_model)
logger.info('Model is loaded.')

def get_category_from_url(url):
    splitted_url = url.split('/')
    categories = splitted_url[3:-1]
    response = {}
    response['url'] = url
    for domain, subdomains in expected_domains.items():
        if not categories:
            for category in categories:
                if any(category == subdomain for subdomain in subdomains):
                    response['category'] = domain
                    return response
        else:
            if any(subdomain in url for subdomain in subdomains):
                response['category'] = domain
                return response
    response['category'] = "unknown"
    return response


def get_category_from_content(content):
    '''
    Only supports ['business', 'entertainment', 'politics', 'sport', 'tech']
    :param content:
    :type content:
    :return:
    :rtype:
    '''
    content = content.lower()
    logger.debug("Content {}".format(content))
    prediction = str(model.predict(content))
    logger.debug("Predicted category is {}".format(prediction))
    return {'content': content, 'category': prediction}
