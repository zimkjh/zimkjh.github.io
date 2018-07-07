---
layout: page
permalink: /about/index.html
title: JoHyun Kim
tags: [JoHyun]
imagefeature: ham.png
chart: true
---
<figure>
  <img src="{{ site.url }}/images/ham_photo.png" alt="JoHyun Kim">
<!--   <figcaption>Hossain Mohammad Faysal</figcaption> -->
</figure>

{% assign total_words = 0 %}
{% assign total_readtime = 0 %}
{% assign featuredcount = 0 %}
{% assign statuscount = 0 %}

{% for post in site.posts %}
    {% assign post_words = post.content | strip_html | number_of_words %}
    {% assign readtime = post_words | append: '.0' | divided_by:200 %}
    {% assign total_words = total_words | plus: post_words %}
    {% assign total_readtime = total_readtime | plus: readtime %}
    {% if post.featured %}
    {% assign featuredcount = featuredcount | plus: 1 %}
    {% endif %}
{% endfor %}


연세대학교 컴퓨터과학과

연세대학교 빅데이터학회 YBIGTA 11기

***This is the space to create.***
