import unittest
from web_app.web_parser import GovUKParser

class TestGovUKParser(unittest.TestCase):

    def test_basic_govspeak_extraction(self):
        html = '<html><body><div class="govspeak"><p>This is the main content.</p><p>Second paragraph.</p></div></body></html>'
        parser = GovUKParser(html_content=html)
        raw_text, clean_text = parser.extract_govspeak_text()
        self.assertIn("This is the main content", raw_text)
        self.assertIn("Second paragraph", clean_text)

    def test_footnote_removal(self):
        html = '<html><body><div class="govspeak"><p>Content [footnote 1] here.</p></div></body></html>'
        parser = GovUKParser(html_content=html)
        raw_text, clean_text = parser.extract_govspeak_text()
        self.assertIn("[footnote 1]", raw_text)
        self.assertNotIn("[footnote 1]", clean_text)
        self.assertIn("Content here", clean_text)

    def test_footnotes_extraction(self):
        html = '''
        <html><body>
            <div class="govspeak"><p>Content</p></div>
            <div class="footnotes">
                <div id="fn:1">First footnote</div>
                <div id="fn:2">Second footnote</div>
            </div>
        </body></html>
        '''
        parser = GovUKParser(html_content=html)
        footnotes = parser.extract_footnotes()
        self.assertEqual(len(footnotes), 2)
        self.assertEqual(footnotes["1"]["text"], "First footnote")

    def test_footnotes_extraction_urls(self):
        html = '''
        <html><body>
            <div class="govspeak"><p>Content</p></div>
            <div class="footnotes">
                <div id="fn:1">See <a href="https://example.com/source">ONS</a> data.
                    <a href="#fnref:1" class="reversefootnote">&#8617;</a>
                </div>
            </div>
        </body></html>
        '''
        parser = GovUKParser(html_content=html)
        footnotes = parser.extract_footnotes()
        self.assertEqual(footnotes["1"]["urls"], ["https://example.com/source"])

    def test_parse_method(self):
        html = '''
        <html><body>
            <div class="govspeak"><p>Content [footnote 1] here.</p></div>
            <div class="footnotes"><div id="fn:1">The footnote</div></div>
        </body></html>
        '''
        parser = GovUKParser(html_content=html)
        results = parser.parse()
        self.assertIn("Content here", results['govspeak_text_clean'])
        self.assertNotIn("[footnote 1]", results['govspeak_text_clean'])
        self.assertEqual(len(results['footnotes']), 1)
        self.assertEqual(results['footnotes']['1']['text'], "The footnote")

    def test_single_reference_no_trailing_bold(self):
        html = '<html><body><div class="govspeak"><p>Content here<sup><a href="#fn:1">1</a></sup>.</p></div></body></html>'
        parser = GovUKParser(html_content=html)
        _, clean_text = parser.extract_govspeak_text()
        self.assertIn("** (Reference: 1)", clean_text)
        self.assertNotIn("** (Reference: 1)**", clean_text)

    def test_multiple_references_same_sentence_merged(self):
        html = (
            '<html><body><div class="govspeak">'
            '<p>Sales rose<sup><a href="#fn:1">1</a></sup> due to demand'
            '<sup><a href="#fn:2">2</a></sup>. A new sentence follows.</p>'
            '</div></body></html>'
        )
        parser = GovUKParser(html_content=html)
        _, clean_text = parser.extract_govspeak_text()
        self.assertIn("** (Reference: 1, 2)", clean_text)
        self.assertIn("A new sentence follows.", clean_text)

    def test_body_links_annotated_and_recorded(self):
        html = (
            '<html><body><div class="govspeak">'
            '<p>See the <a href="https://www.gov.uk/guidance">guidance</a> '
            'and the <a href="https://example.com/report">report</a> for more.</p>'
            '</div></body></html>'
        )
        parser = GovUKParser(html_content=html)
        _, clean_text = parser.extract_govspeak_text()
        self.assertIn("guidance**(Link: 1)", clean_text)
        self.assertIn("report**(Link: 2)", clean_text)

        results = parser.parse()
        self.assertEqual(len(results['links']), 2)
        self.assertEqual(results['links'][0], {
            'number': '1', 'url': 'https://www.gov.uk/guidance', 'text': 'guidance'
        })
        self.assertEqual(results['links'][1]['url'], 'https://example.com/report')

    def test_body_links_skip_in_page_anchors(self):
        html = (
            '<html><body><div class="govspeak">'
            '<p>Jump to the <a href="#contents">contents</a> section.</p>'
            '</div></body></html>'
        )
        parser = GovUKParser(html_content=html)
        _, clean_text = parser.extract_govspeak_text()
        self.assertNotIn("Link:", clean_text)
        self.assertEqual(parser.links, [])

if __name__ == '__main__':
    unittest.main()