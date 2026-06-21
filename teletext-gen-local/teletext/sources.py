from dataclasses import dataclass
from typing import Tuple


@dataclass
class TeletextSource:
    name: str
    display_name: str
    base_url: str
    page_range: Tuple[int, int]
    subpage_range: Tuple[int, int]
    charset: str
    language: str
    image_format: str
    notes: str = ''


SOURCES: dict = {

    'hrt': TeletextSource(
        name='hrt',
        display_name='HRT (Croatia)',
        base_url='https://teletekst.hrt.hr/{page}-{subpage:02d}.HTML',
        page_range=(100, 899),
        subpage_range=(1, 8),
        charset='croatian',
        language='hr',
        image_format='base64_gif',
    ),

    'rtvslo': TeletextSource(
        name='rtvslo',
        display_name='RTVSLO (Slovenia)',
        base_url='https://teletext.rtvslo.si/ttxdata/{page}_{subpage:04d}.png',
        page_range=(100, 899),
        subpage_range=(1, 8),
        charset='slovenian',
        language='sl',
        image_format='direct_url',
        notes='480x336 PNG via /ttxdata/{page}_{subpage:04d}.png. No HTML page needed.',
    ),

    'rtvfbih': TeletextSource(
        name='rtvfbih',
        display_name='RTVFBiH (Bosnia)',
        base_url='https://teletext.rtvfbih.ba/{page}/{page}_{subpage:04d}.png',
        page_range=(100, 899),
        subpage_range=(1, 8),
        charset='bosnian',
        language='bs',
        image_format='direct_url',
        notes='480x336 PNG via frameset: {page}/{page}_{subpage:04d}.htm contains the image ref.',
    ),

    'nos': TeletextSource(
        name='nos',
        display_name='NOS (Netherlands)',
        base_url='https://teletekst.nos.nl/{page}',
        page_range=(100, 899),
        subpage_range=(1, 4),
        charset='dutch',
        language='nl',
        image_format='nextjs_spa',
        notes='Next.js SPA with hash routing. Data embedded in __NEXT_DATA__ as HTML+CSS spans, not images. Needs custom parser.',
    ),

    'orf': TeletextSource(
        name='orf',
        display_name='ORF (Austria)',
        base_url='https://appmeta.orf.at/teletext/orf1/{page}_{subpage:04d}.png',
        page_range=(100, 899),
        subpage_range=(1, 8),
        charset='german',
        language='de',
        image_format='direct_url',
        notes='720x432 PNG. Change orf1 to orf2 in base_url for channel 2. orf3 returns 403.',
    ),

    'ct': TeletextSource(
        name='ct',
        display_name='Czech Television (Czech Republic)',
        base_url='https://api-teletext.ceskatelevize.cz/pages/{ct_page}/image.webp',
        page_range=(100, 899),
        subpage_range=(1, 8),
        charset='czech',
        language='cs',
        image_format='direct_url',
        notes='Direct WebP image endpoint. Web page p=100-1 resolves to API page 100A.',
    ),

    'svt': TeletextSource(
        name='svt',
        display_name='SVT (Sweden)',
        base_url='https://www.svt.se/text-tv/{page}',
        page_range=(100, 899),
        subpage_range=(1, 4),
        charset='swedish',
        language='sv',
        image_format='base64_gif',
        notes='Confirmed: base64 GIF embedded in HTML. URL path is /text-tv/ not /teletext/.',
    ),

    'rtp': TeletextSource(
        name='rtp',
        display_name='RTP (Portugal)',
        base_url='https://www.rtp.pt/wportal/teletexto/{page}/{page}_{subpage:04d}.png',
        page_range=(100, 899),
        subpage_range=(1, 4),
        charset='portuguese',
        language='pt',
        image_format='direct_url',
        notes='480x336 PNG. Text-only version also available at /wportal/teletexto/texto/?p={page}',
    ),
}


def ct_subpage_suffix(subpage: int) -> str:
    """Convert CT subpage number to API suffix: 1 -> A, 2 -> B, etc."""
    if subpage < 1 or subpage > 26:
        raise ValueError(f"CT subpage must be in 1..26, got {subpage}")
    return chr(ord('A') + subpage - 1)
